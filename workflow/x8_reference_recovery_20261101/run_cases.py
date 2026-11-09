"""Transparent phase recording around the unchanged nominal particle solver."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import copy
import fcntl
import json
import multiprocessing
import os
import time

import h5py
import numpy as np
import shared as s
import worker
import engine

EVENTS=None
PHASE='chemistry'
FINAL=None
FINAL_DAMPING=None
BASE_SOLVER=None
LINES=('co','13co','c18o')


def wrap_emitter(previous):
    def observed(self,*args,**kwargs):
        result=previous(self,*args,**kwargs)
        if EVENTS is not None:
            name=str(self.name).lower()
            EVENTS[(PHASE,name,'fail' if result is False else 'ok')]+=1
            if PHASE=='capture' and name in LINES:
                FINAL[name]=int(result is True)
                FINAL_DAMPING[name]=float(kwargs.get('dampFactor') if kwargs.get('dampFactor') is not None else self.dampFactor)
        return result
    return observed


def wrap_capture(previous):
    def capture(model):
        global PHASE
        PHASE='capture'
        try:return previous(model)
        finally:PHASE='chemistry'
    return capture


def initialize():
    global BASE_SOLVER
    worker.initialize()
    BASE_SOLVER=copy.deepcopy(worker.DIRECT.SOLVER)
    assert BASE_SOLVER['tEqGuess_s'] is None and BASE_SOLVER['maximum_time_retry_s']==1e18
    from despotic.emitter import emitter
    emitter.setLevPopEscapeProb=wrap_emitter(emitter.setLevPopEscapeProb)
    worker.capture=wrap_capture(worker.capture)


def solve(task):
    global EVENTS,PHASE,FINAL,FINAL_DAMPING
    index,mode,branch,row=task
    EVENTS=Counter();PHASE='chemistry';FINAL={};FINAL_DAMPING={}
    worker.DIRECT.SOLVER=copy.deepcopy(BASE_SOLVER)
    worker.DIRECT.SOLVER['maximum_time_primary_s']=1e18 if mode==0 else 1e16
    result=worker.solve((index,branch,row))
    final=np.array([FINAL.get(name,-1) for name in LINES],dtype='i1')
    damping=np.array([FINAL_DAMPING.get(name,np.nan) for name in LINES])
    events=json.dumps({'|'.join(key):value for key,value in EVENTS.items()},sort_keys=True)
    assert len(events.encode())<4096
    return result,final,damping,events


def main():
    assert int(os.environ.get('PBS_NCPUS','0'))>0
    sha=s.verify();part=int(os.environ['PARTITION']);assert 0<=part<8
    with h5py.File(s.ROOT/'cases.h5') as f:
        ids=np.arange(part,len(f['state_id']),8);state_id=f['state_id'][:][ids]
        modes=f['mode'][:][ids];branches=f['branch'][:][ids];raw=f['raw'][:][ids]
    target=s.ROOT/f'output/part{part:02d}.npz'
    lock=(s.ROOT/f'output/part{part:02d}.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    meta=dict(manifest_sha256=sha,partition=part)
    state=s.base.core.empty_state(len(ids));final=np.full((len(ids),3),-1,'i1');damping=np.full((len(ids),3),np.nan)
    events=np.full(len(ids),b'',dtype='S4096')
    if target.exists():
        with np.load(target,allow_pickle=False) as f:
            assert json.loads(str(f['metadata']))==meta
            np.testing.assert_array_equal(f['case_ids'],ids)
            state={k:f[k].copy() for k in state};final=f['final_returns'].copy();damping=f['final_damping'].copy();events=f['events'].copy()
    started=time.monotonic()
    def checkpoint():
        complete=bool(np.all(state['status']!=0))
        s.base.atomic_npz(target,metadata=np.asarray(json.dumps(meta,sort_keys=True)),case_ids=ids,state_id=state_id,
            mode=modes,final_returns=final,final_damping=damping,events=events,complete=np.asarray(complete),**state)
        s.save(s.ROOT/f'output/part{part:02d}.json',dict(partition=part,complete=complete,total=len(ids),
            pending=int(np.sum(state['status']==0)),success=int(np.sum(state['status']==1)),
            failed=int(np.sum(state['status']==-1)),timeout=int(np.sum(state['status']==-2)),elapsed_seconds=time.monotonic()-started))
    try:
        with ProcessPoolExecutor(max_workers=min(104,int(os.environ['PBS_NCPUS'])),
                mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
            futures={pool.submit(solve,(int(i),int(modes[i]),s.base.BRANCHES[branches[i]],tuple(raw[i]))):int(i)
                for i in np.flatnonzero(state['status']==0)}
            for done,future in enumerate(as_completed(futures),1):
                result,returns,df,ev=future.result();i=futures[future];assert result[0]==i
                engine.apply_result(state,result);final[i]=returns;damping[i]=df;events[i]=ev.encode()
                if done%128==0:checkpoint();print('recovery partition',part,done,flush=True)
    finally:checkpoint();lock.close()
    assert np.all(state['status']!=0)


if __name__=='__main__':main()
