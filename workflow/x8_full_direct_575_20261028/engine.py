"""Whole-population lanes with immutable block results and atomic checkpoints."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import fcntl
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import time

import numpy as np
from core import (ROOT,load_plan,verify_code,read_task,identity,paths,empty_state,
    atomic_state,read_state,save_json,digest)
import worker

STOP=False


def stop_handler(*_):
    global STOP
    STOP=True


def apply_result(state,result):
    i,status,attempts,seconds,calls,failures,flags,values,chemistry,error=result
    assert state['status'][i]==0,'Attempt to overwrite a committed physical state'
    state['values'][i]=values; state['chemistry'][i]=chemistry
    state['attempts'][i]=attempts; state['seconds'][i]=seconds
    state['ode_calls'][i]=calls; state['ode_failures'][i]=failures
    state['quality_flags'][i]=flags; state['error'][i]=error.encode()[:768]
    state['status'][i]=status


def process_block(pool,task,plan,audit,root=ROOT,stop_after=None):
    data=read_task(task,plan); metadata=identity(task,data,plan,audit); n=len(data['raw'])
    final,checkpoint=paths(task,root)
    if final.exists():
        state=read_state(final,metadata,n,require_complete=True,data=data)
        return True,state
    state=read_state(checkpoint,metadata,n,data=data) if checkpoint.exists() else empty_state(n)
    todo=np.flatnonzero(state['status']==0)
    futures={pool.submit(worker.solve,(int(i),task['branch'],tuple(data['raw'][i]))):int(i) for i in todo}
    completed=0
    try:
        for future in as_completed(futures):
            if STOP: break
            result=future.result(); assert result[0]==futures[future]
            apply_result(state,result); completed+=1
            if completed%128==0: atomic_state(checkpoint,metadata,state,data,False)
            if stop_after is not None and completed>=stop_after: break
    finally:
        for future in futures: future.cancel()
        complete=bool(np.all(state['status']!=0))
        atomic_state(final if complete else checkpoint,metadata,state,data,complete)
        if complete and checkpoint.exists(): checkpoint.unlink()
    return complete,state


def main():
    global STOP
    assert int(os.environ.get('PBS_NCPUS','0'))>0
    lane=int(os.environ['LANE']); assert 0<=lane<4
    sha=verify_code(); plan=load_plan(); audit=json.loads((ROOT/'input_audit.json').read_text())
    assert audit['manifest_sha256']==sha and audit['passed']
    assert digest(ROOT/'preflight.json')==audit['preflight_sha256']
    lockpath=ROOT/f'lanes/lane{lane:02d}.lock'
    lock=lockpath.open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lane_status=ROOT/f'lanes/lane{lane:02d}.json'
    status=dict(lane=lane,job_id=os.environ.get('PBS_JOBID'),manifest_sha256=sha,
        complete=False,completed_blocks=0,states=0,success=0,failed=0,timeouts=0)
    # A cursor is an optimization only; every prior final result is revalidated
    # when this lane resumes, so stale/missing progress JSON cannot hide holes.
    workers=min(104,int(os.environ['PBS_NCPUS']))
    started=time.monotonic(); deadline=started+46*3600
    signal.signal(signal.SIGTERM,stop_handler); signal.signal(signal.SIGINT,stop_handler)
    save_json(lane_status,status)
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=worker.initialize)
    assigned=[task for task in plan['tasks'] if task['id']%4==lane]
    completed_all=False
    try:
        for task in assigned:
            final,_=paths(task)
            worst=math.ceil((task['stop']-task['start'])/workers)*180+120
            if not final.exists() and (STOP or time.monotonic()+worst>deadline):
                completed_all=False; break
            complete,state=process_block(pool,task,plan,audit)
            if not complete:
                completed_all=False; break
            status['completed_blocks']+=1
            status['states']+=len(state['status'])
            status['success']+=int(np.sum(state['status']==1))
            status['failed']+=int(np.sum(state['status']==-1))
            status['timeouts']+=int(np.sum(state['status']==-2))
            status.update(last_task=task['id'],elapsed_seconds=time.monotonic()-started)
            save_json(lane_status,status)
            print(json.dumps(status),flush=True)
        else:
            completed_all=True
    finally:
        pool.shutdown(wait=True,cancel_futures=True)
        status.update(complete=bool(completed_all),elapsed_seconds=time.monotonic()-started,
            continuation_required=not completed_all,assigned_blocks=len(assigned))
        save_json(lane_status,status)
        fcntl.flock(lock,fcntl.LOCK_UN); lock.close()


if __name__=='__main__': main()
