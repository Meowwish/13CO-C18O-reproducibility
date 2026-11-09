"""Finite-population, bootstrap, mask and process-identity checks; no chemistry."""
from concurrent.futures import ProcessPoolExecutor
import ast
import importlib.util
import itertools
import multiprocessing

import numpy as np

import selection_common as c
from selection_design import allocate,total_and_covariance,median_bounds,selected_bounds,difference_bounds
from selection_statistics import median,detection
from selection_bootstrap import replicate_weights
from integration_check import run as synthetic_integration


def spawned_imports():
    import selection_prepare as prepare
    import selection_solve as solve
    import selection_review as review
    for module in (prepare,solve,review):assert str(c.ROOT) in module.__file__
    assert callable(solve.worker.solve) and solve.c.PARTITIONS==64
    assert solve.worker.c.PROTOCOLS[1]=='mxstep5000_600s'
    return True


def main():
    for path in c.ROOT.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    actual=np.array([[1.,2.],[4.,1.],[2.,7.],[9.,4.],[2.,4.],[6.,2.],[4.,3.]])
    proxy=np.array([[2.,2.],[3.,2.],[3.,4.],[7.,3.],[3.,3.],[5.,3.],[4.,3.]])
    # Include a failed direct target and its matched proxy exclusion as four
    # coupled totals, rather than testing only an ordinary mean.
    success=np.array([1,1,0,1,0,1,1],bool)
    residual=np.column_stack((np.where(success[:,None],actual,0.)-proxy,np.where(success[:,None],0.,-proxy)))
    baseline=np.r_[proxy.sum(axis=0),proxy.sum(axis=0)]
    points=[];covariances=[];population=np.array([4,3])
    for one in itertools.combinations(range(4),2):
        for two in itertools.combinations(range(4,7),2):
            ids=np.array(one+two);sh=np.array([0,0,1,1]);pi=np.array([.5,.5,2/3,2/3])
            t,v=total_and_covariance(baseline,residual[ids],pi,sh,population)
            points.append(t);covariances.append(v)
    truth=np.r_[actual[success].sum(axis=0),proxy[success].sum(axis=0)]
    np.testing.assert_allclose(np.mean(points,axis=0),truth,atol=1e-12)
    np.testing.assert_allclose(np.mean(covariances,axis=0),np.cov(points,rowvar=False,ddof=0),atol=1e-12)
    alloc=allocate(np.array([1,3,20,0]),np.array([0.,0.,5.,0.]),12)
    assert alloc.sum()==12 and np.all(alloc<=[1,3,20,0]) and np.all(alloc[[0,1,2]]>0)
    low=np.array([1.,2.,1.5,5.,3.]);high=np.array([3.,4.,6.,7.,8.]);weight=np.array([1.,3.,2.,1.,2.])
    mandatory=np.array([1,1,0,0,0],bool);possible=np.ones(5,bool);bounds=median_bounds(low,high,weight,mandatory,possible)
    enclosed=0
    for chosen in itertools.product((False,True),repeat=3):
        mask=mandatory.copy();mask[2:]=chosen
        for ends in itertools.product((False,True),repeat=5):
            value=np.where(ends,high,low);m=median(value[mask],weight[mask])
            assert bounds[0]<=m<=bounds[1];enclosed+=1
    line=np.array([[4.,1.],[12.,3.],[.01,.001]])
    exact=selected_bounds(line,line,np.array([1.,2.,3.]),1,'dual')
    np.testing.assert_allclose(exact[:3],np.full((3,2),4.))
    np.testing.assert_allclose(difference_bounds(exact,exact),np.zeros((4,2)),atol=1e-12)
    # Enumerate every multinomial resample for n=3 (resample size n-1=2).
    class Exhaustive:
        def multinomial(self,n,p,size):
            assert n==2 and len(p)==3 and size==9
            return np.array([np.bincount(x,minlength=3) for x in itertools.product(range(3),repeat=2)])
    bw=replicate_weights(np.zeros(3,'i8'),np.array([7]),Exhaustive(),9)
    x=np.array([[1.,4.],[4.,1.],[9.,6.]])
    totals=(bw*(7/3))@x
    np.testing.assert_allclose(totals.mean(axis=0),(7/3)*x.sum(axis=0),atol=1e-12)
    target=7**2*(1-3/7)/3*np.cov(x,rowvar=False,ddof=1)
    np.testing.assert_allclose(np.cov(totals,rowvar=False,ddof=0),target,atol=1e-12)
    censused=replicate_weights(np.zeros(3,'i8'),np.array([3]),np.random.default_rng(3),20)
    np.testing.assert_array_equal(censused,np.ones((20,3)))
    # All 3^5 unknown/success/failure configurations: omitted states cannot
    # alter a scientific target, regardless of their hypothetical outcomes.
    configurations=0;outcomes=0
    for state in itertools.product((-1,0,1),repeat=5):
        status=np.array([state],'i1');q=status==1;need=c.needed_states(status,q)[0]
        missing=np.flatnonzero(status[0]==0)
        for result in itertools.product((-1,1),repeat=len(missing)):
            truth=status.copy();truth[0,missing]=result
            partial=status.copy();partial[0,need]=truth[0,need]
            assert c.known_models(partial,partial==1).all()
            truth_q=truth==1;partial_q=partial==1
            a=np.r_[truth_q[0,:2],np.repeat(truth_q[0,[0,2,3,4]].all(),4)]
            b=np.r_[partial_q[0,:2],np.repeat(partial_q[0,[0,2,3,4]].all(),4)]
            np.testing.assert_array_equal(a,b);outcomes+=1
        configurations+=1
    spec=importlib.util.spec_from_file_location('old_selection',c.OLD/'aperture_statistics.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    mask_checks=0
    with np.load(c.PILOT/'aperture_results.npz') as f:
        for field in ('estimates','reference_low','reference_high'):
            for cols in (slice(0,2),slice(2,4)):
                for m in range(6):
                    for depth in c.DEPTHS:
                        line=f[field][:,m,cols]
                        with np.errstate(over='ignore'):expected=old.detection(line,depth,'dual_and_ratio_snr')
                        np.testing.assert_array_equal(detection(line,depth,'dual_and_ratio_snr'),expected)
                        mask_checks+=1
    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
        assert pool.submit(spawned_imports).result(timeout=90)
        integration=pool.submit(synthetic_integration).result(timeout=90)
    assert integration['passed']
    c.save(c.ROOT/'local_checks.json',dict(passed=True,exhaustive_stratified_draws=len(points),
        unbiased_direct_and_matched_totals=True,joint_covariance_matches=True,
        catalogue_configurations_enclosed=enclosed,rescaled_bootstrap_enumerations=9,
        rescaled_bootstrap_covariance_matches=True,census_stratum_variance_zero=True,
        branch_status_configurations=configurations,branch_outcomes_checked=outcomes,
        overflow_free_mask_comparisons=mask_checks,spawned_module_identities=True,
        synthetic_prepare_review_integration=integration,chemical_solves_on_login=0))
    print('Finite design, selection, census-aware bootstrap and spawned-module tests passed.',flush=True)


if __name__=='__main__':main()
