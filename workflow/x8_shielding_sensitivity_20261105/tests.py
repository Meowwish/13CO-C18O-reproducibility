"""No cloud solves on login: imports, lookup routing, probability and scores."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os

import h5py
import numpy as np

import shield_common as c


def imports():
    import solve_shielding
    import workflow_config
    from despotic.chemistry.GOW import GOW
    return callable(solve_shielding.solve) and callable(GOW.dxdt_type)


def main():
    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
        assert pool.submit(imports).result(timeout=45)
    import shield_patch as patch
    from despotic.chemistry.visser_shielding import VisserCOShielding,fShield_CO_Visser,fShield_13CO_Visser,fShield_C18O_Visser
    grid=VisserCOShielding();original=VisserCOShielding.factor
    expected={v:np.array([original(grid,s,1e16,1e21,c12_c13=v) for s in ('12C16O','13C16O','12C18O')]) for v in (35.,50.,69.,103.)}
    patch.install()
    funcs=(fShield_CO_Visser,fShield_13CO_Visser,fShield_C18O_Visser)
    def call(element):return np.array([fn(1e16,1e21,c12_c13=element) for fn in funcs])
    for mode,ratio in ((0,103.),(1,35.),(2,69.)):
        patch.reset(mode);np.testing.assert_array_equal(call(103.),expected[ratio]);assert patch.COUNTS[0]==3
    np.testing.assert_array_equal(expected[103.],expected[69.])
    patch.reset(3)
    with patch.state_context([50.,1.],0,1):np.testing.assert_array_equal(call(103.),expected[50.])
    assert patch.CURRENT is None and patch.COUNTS[1]==3
    for xin in ([0.,0.],[1.,0.],[1.,1e-320]):
        with patch.state_context(xin,0,1):np.testing.assert_array_equal(call(103.),expected[103.])
    try:
        with patch.state_context([35.,1.],0,1):raise ValueError('context-reset check')
    except ValueError:pass
    assert patch.CURRENT is None
    patch.reset(0);np.testing.assert_array_equal(call(50.),expected[50.])
    rng=np.random.default_rng(c.SEED)
    allocations={}
    for name,folder,count,targets in [('global',c.SOURCE,8192,None),('outer',c.OUTER,4096,[64]*8+[448]*8)]:
        with h5py.File(folder/'sample.h5') as f:
            strata,pop=f['sample/stratum'][:],f['stratum_population'][:]
        ids,allocation=c.nested(strata,pop,count,rng,targets)
        assert len(np.unique(ids))==count and np.all(allocation<=np.bincount(strata,minlength=len(pop)))
        if targets is not None:np.testing.assert_array_equal(np.add.reduceat(allocation,np.arange(0,len(pop),30)),targets)
        allocations[name]=int(allocation.sum())
    # Finite-difference influence derivatives independently test paired sums.
    values=rng.uniform(.1,3,(40,2));mask=np.ones(40,bool);mask[2]=False;values[2]=np.nan
    k=rng.uniform(0,1,(40,4));weight=rng.uniform(1,3,40)
    point,score=c.ratio(values,mask,k,weight)
    for i in (0,2,20):
        changed=weight.copy();changed[i]*=1+1e-5
        p,_=c.ratio(values,mask,k,changed)
        np.testing.assert_allclose((p-point)/1e-5,score[i],atol=1e-10,rtol=2e-5)
    # Avoid the mixed-advanced-indexing regressions found in prior work.
    q=np.ones((17,4,5),bool)
    assert np.all(q[:,2,[0,2,3,4]],axis=1).shape==(17,)
    c.save(c.ROOT/'local_checks.json',dict(passed=True,spawned_imports=True,lookup_routing=True,
        dynamic_context_reset=True,baseline_boundary_identity=True,nested_allocations=allocations,
        finite_difference_scores=True,cloud_solves_performed=0,physical_timescale_work=False))
    print('Shielding preflight passed without any cloud solve.',flush=True)


if __name__=='__main__':main()
