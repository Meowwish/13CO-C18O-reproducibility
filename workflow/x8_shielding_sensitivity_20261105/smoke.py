"""Gate bulk physics on recomputed baseline and exact-identity controls."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
import os

import numpy as np

import shield_common as c
from solve_shielding import initialize,solve,load,raw_for


def main():
    assert int(os.environ.get('PBS_NCPUS','0'))>0
    digest,_=c.ready();assert not (c.ROOT/'smoke.json').exists()
    baseline,qualified,raw=load()
    eligible=np.flatnonzero(qualified.all(1)&np.all(baseline[:,:,6:9]>0,axis=(1,2)))
    assert len(eligible)>=4
    # Rank controls by input density; controls are not used for inference.
    eligible=eligible[np.argsort(raw['actual'][eligible,0])]
    particles=eligible[np.linspace(0,len(eligible)-1,6,dtype=int)[1:5]]
    tasks=[(i,m,b,tuple(raw_for(raw,p,b))) for i,(p,m,b) in enumerate(
        (p,m,b) for p in particles for m in range(4) for b in range(5))]
    values=np.full((4,4,5,27),np.nan);counts=np.zeros((4,4,5,3),'i8')
    qualified_new=np.zeros((4,4,5),bool)
    with ProcessPoolExecutor(max_workers=min(104,int(os.environ['PBS_NCPUS'])),
            mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
        futures=[pool.submit(solve,task) for task in tasks]
        for future in as_completed(futures):
            result,returns,damping,events,trace,lookup=future.result()
            index=result[0];p,m,b=index//20,(index%20)//5,index%5
            values[p,m,b]=result[7];counts[p,m,b]=lookup
            qualified_new[p,m,b]=result[1]==1 and np.all(returns==1)
    c.base.atomic_npz(c.ROOT/'smoke_outputs.npz',particles=particles,values=values,
        qualified=qualified_new,lookup_counts=counts,manifest_sha256=np.asarray(digest))
    passed=bool(qualified_new.all());baseline_error=None;identity_error=None
    if passed:
        baseline_error=float(np.max(np.abs(np.log10(values[:,0,:,6:9]/baseline[particles,:,6:9]))))
        identity_error=float(np.max(np.abs(np.log10(values[:,2,3:5,6:9]/values[:,0,3:5,6:9]))))
        passed=baseline_error<=.02 and identity_error<=1e-10 and bool(np.all(counts[:,3,:,1]>0))
    c.save(c.ROOT/'smoke.json',dict(passed=bool(passed),manifest_sha256=digest,
        particles=particles,qualified_states=int(qualified_new.sum()),total_states=80,
        maximum_baseline_line_error_dex=baseline_error,maximum_uniform_fixed69_identity_error_dex=identity_error))
    assert passed,'Smoke gate failed; bulk dependencies must not run'
    print('80-state compute-node smoke gate passed.',flush=True)


if __name__=='__main__':main()
