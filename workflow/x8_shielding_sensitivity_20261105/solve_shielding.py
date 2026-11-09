"""Strict paired solves with exact-identity reuse and resumable partitions."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import fcntl
import json
import multiprocessing
import os

import h5py
import numpy as np

import shield_common as c
import shield_patch as patch
import solve_diagnostics as runtime
import engine


def initialize():
    runtime.initialize();patch.install()


def solve(task):
    index,mode,branch,raw=task
    patch.reset(mode)
    result=runtime.solve((index,1,c.BRANCHES[branch],raw))
    return (*result,patch.COUNTS.copy())


def load():
    with h5py.File(c.ROOT/'sample.h5') as f:
        values=np.concatenate([f[name+'/baseline_values'][:] for name in ('global','outer')])
        qualified=np.concatenate([f[name+'/baseline_qualified'][:] for name in ('global','outer')])
        raw={key:np.concatenate([f[name+'/'+key][:] for name in ('global','outer')]) for key in ('actual','slug','uniform')}
    return values,qualified,raw


def raw_for(raw,particle,branch):
    key='slug' if branch==1 else 'uniform' if branch>=3 else 'actual'
    return raw[key][particle]


def main():
    assert int(os.environ.get('PBS_NCPUS','0'))>0
    digest,ready=c.ready();assert json.loads((c.ROOT/'smoke.json').read_text())['passed']
    part=int(os.environ['PARTITION']);assert 0<=part<c.PARTITIONS
    baseline,qualified,raw=load();ids=np.arange(part,ready['logical_states'],c.PARTITIONS)
    particle,mode,branch=ids//20,(ids%20)//5,ids%5
    path=c.ROOT/f'direct/part{part:02d}.npz'
    lock=path.with_suffix('.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=c.base.core.empty_state(len(ids));final=np.full((len(ids),3),-1,'i1')
    origin=np.zeros(len(ids),'u1');counts=np.zeros((len(ids),3),'i8')
    if path.exists():
        with np.load(path) as f:
            assert str(f['manifest_sha256'])==digest;np.testing.assert_array_equal(f['task_ids'],ids)
            for key in state:state[key]=f[key].copy()
            final,origin,counts=[f[key].copy() for key in ('final_returns','origin','lookup_counts')]
    else:
        for i,(p,m,b) in enumerate(zip(particle,mode,branch)):
            row=raw_for(raw,p,b);ratio=row[3]/row[4] if row[4]>0 else np.inf
            identical=m==0 or (m==1 and ratio<=35) or (m==2 and ratio>=69)
            if identical and qualified[p,b]:
                state['status'][i]=1;state['values'][i]=baseline[p,b];final[i]=1;origin[i]=1
    def checkpoint():
        q=(state['status']==1)&np.all(final==1,axis=1)
        complete=bool(np.all(state['status']!=0))
        c.base.atomic_npz(path,manifest_sha256=np.asarray(digest),sample_sha256=np.asarray(ready['sample_sha256']),
            task_ids=ids,complete=np.asarray(complete),qualified=q,final_returns=final,
            origin=origin,lookup_counts=counts,**state)
        c.save(path.with_suffix('.json'),dict(complete=complete,total=len(ids),pending=int(np.sum(state['status']==0)),
            qualified=int(q.sum()),failed=int(np.sum(state['status']==-1)),timeout=int(np.sum(state['status']==-2)),
            reused=int(np.sum(origin==1))))
    checkpoint()
    try:
        with ProcessPoolExecutor(max_workers=min(104,int(os.environ['PBS_NCPUS'])),
                mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
            futures={pool.submit(solve,(int(i),int(mode[i]),int(branch[i]),tuple(raw_for(raw,particle[i],branch[i])))):int(i)
                for i in np.flatnonzero(state['status']==0)}
            for done,future in enumerate(as_completed(futures),1):
                result,returns,damping,events,trace,lookup=future.result();i=futures[future]
                assert result[0]==i;engine.apply_result(state,result)
                final[i]=returns;origin[i]=2;counts[i]=lookup
                if done%64==0:checkpoint();print('shield partition',part,done,'/',len(futures),flush=True)
    finally:checkpoint();lock.close()
    assert np.all(state['status']!=0),'Explicit resume required for pending states'


if __name__=='__main__':main()
