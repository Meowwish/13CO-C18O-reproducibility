"""Disjoint sample-state partitions using the unchanged full-run worker."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import fcntl
import json
import multiprocessing
import os
import signal
import time

import h5py
import numpy as np
import common as c
import engine
import worker

STOP=False


def stop(*_):
    global STOP
    STOP=True


def main():
    assert int(os.environ.get('PBS_NCPUS','0'))>0
    sha,ready=c.ready();part=int(os.environ['PARTITION']);assert 0<=part<c.PARTITIONS
    target=c.ROOT/f'direct/part{part:02d}.npz'
    lock=(c.ROOT/f'direct/part{part:02d}.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with h5py.File(c.ROOT/'sample.h5') as f:
        n=len(f['sample/mass']);ids=np.arange(part,n*5,8);uid=ids//5;branch=ids%5
        raw={k:f['sample/'+k][:] for k in ('actual','slug','uniform')}
        particle_id=f['sample/particle_id'][:][uid];snapshot_row=f['sample/row'][:][uid]
        old={k:f['reuse_actual/'+k][:] for k in c.core.empty_state(0)}
    metadata=dict(manifest_sha256=sha,sample_sha256=ready['sample_sha256'],partition=part,partitions=8)
    state=c.core.empty_state(len(ids));origin=np.zeros(len(ids),'u1')
    if target.exists():
        with np.load(target,allow_pickle=False) as f:
            assert json.loads(str(f['metadata']))==metadata
            np.testing.assert_array_equal(f['task_ids'],ids)
            np.testing.assert_array_equal(f['particle_id'],particle_id)
            state={k:f[k].copy() for k in state};origin=f['origin'].copy()
    else:
        reused=(branch==0)&(old['status'][uid]!=0)
        for key in state:state[key][reused]=old[key][uid[reused]]
        origin[reused]=1
    assert np.all(np.isin(state['status'],[0,1,-1,-2]))
    assert np.all(np.isnan(state['values'][state['status']<=0]))
    assert np.all(np.isfinite(state['values'][state['status']==1]))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    started=time.monotonic();pool=None
    def checkpoint():
        complete=bool(np.all(state['status']!=0))
        c.atomic_npz(target,metadata=np.asarray(json.dumps(metadata,sort_keys=True)),task_ids=ids,
            particle_id=particle_id,snapshot_row=snapshot_row,origin=origin,complete=np.asarray(complete),**state)
        c.save(c.ROOT/f'direct/part{part:02d}.json',dict(partition=part,job_id=os.environ.get('PBS_JOBID'),
            complete=complete,total=len(ids),pending=int(np.sum(state['status']==0)),
            success=int(np.sum(state['status']==1)),failed=int(np.sum(state['status']==-1)),
            timeout=int(np.sum(state['status']==-2)),reused=int(np.sum(origin==1)),
            newly_attempted=int(np.sum(origin==2)),elapsed_seconds=time.monotonic()-started))
    checkpoint()
    try:
        todo=np.flatnonzero(state['status']==0)
        if len(todo):
            pool=ProcessPoolExecutor(max_workers=min(104,int(os.environ['PBS_NCPUS'])),
                mp_context=multiprocessing.get_context('spawn'),initializer=worker.initialize)
            futures={}
            for i in todo:
                b=c.BRANCHES[branch[i]];key='slug' if b=='slug_on' else 'uniform' if b.startswith('uniform') else 'actual'
                future=pool.submit(worker.solve,(int(i),b,tuple(raw[key][uid[i]])))
                futures[future]=int(i)
            for k,future in enumerate(as_completed(futures),1):
                if STOP:break
                result=future.result();assert result[0]==futures[future]
                engine.apply_result(state,result);origin[result[0]]=2
                if k%256==0:
                    checkpoint();print('solver partition',part,'new states',k,'/',len(todo),flush=True)
            for future in futures:future.cancel()
    finally:
        checkpoint()
        if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
        checkpoint();lock.close()
    assert np.all(state['status']!=0),'Incomplete partition; resume pending labels explicitly'


if __name__=='__main__':main()
