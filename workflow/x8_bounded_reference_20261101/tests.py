"""Small deterministic design/variance/geometry checks; no chemistry on login."""
from itertools import combinations
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
import common as c
from design_statistics import joint_covariance,log_metrics
from geometry import adjoint
from review_sample import assess


def assembled_review():
    """Exercise all eight result files and a real report with synthetic labels."""
    import review_sample
    previous=c.ROOT,c.ready,os.environ.get('PBS_NCPUS')
    try:
        with tempfile.TemporaryDirectory() as tmp:
            c.ROOT=Path(tmp);(c.ROOT/'direct').mkdir();(c.ROOT/'geometry').mkdir()
            c.ready=lambda:('test',dict(sample_sha256='sample'))
            os.environ['PBS_NCPUS']='1'
            n=4;pids=np.arange(10,14)
            xyz=np.column_stack((np.arange(1500,5500,1000),np.zeros(4),np.zeros(4)))
            with h5py.File(c.ROOT/'sample.h5','w') as f:
                for k,v in dict(mass=np.ones(n),nuclei=np.ones(n),xyz=xyz,
                    inclusion_probability=np.full(n,.5),stratum=np.zeros(n,'i8'),particle_id=pids,
                    lum_per_H=np.ones((n,6,3))).items():f.create_dataset('sample/'+k,data=v)
                f.create_dataset('stratum_population',data=np.array([8]));f.create_dataset('stratum_sample',data=np.array([4]))
                for g in ('particle','beam'):f.create_dataset('predicted_'+g+'_totals',data=np.ones((9,6,3))*8)
            for part in range(8):
                ids=np.arange(part,n*5,8);state=c.core.empty_state(len(ids))
                state['status'][:]=1;state['values'][:]=1;state['chemistry'][:]=1
                if part==0:state['status'][0]=-1;state['values'][0]=np.nan;state['chemistry'][0]=np.nan;state['error'][0]=b'test failure'
                c.atomic_npz(c.ROOT/f'direct/part{part:02d}.npz',
                    metadata=np.asarray(json.dumps(dict(manifest_sha256='test',sample_sha256='sample'))),
                    task_ids=ids,particle_id=pids[ids//5],origin=np.full(len(ids),2,'u1'),complete=np.asarray(True),**state)
            factors=c.region_masks(xyz).astype(float)
            for part in range(4):
                c.atomic_npz(c.ROOT/f'geometry/part{part:02d}.npz',manifest_sha256=np.asarray('test'),
                    sample_sha256=np.asarray('sample'),complete=np.asarray(True),ids=np.array([part]),kernel_weights=factors[part:part+1])
            review_sample.main()
            result=json.loads((c.ROOT/'review.json').read_text())
            assert result['branches']['actual_on']['success']==3 and result['branches']['actual_on']['failed']==1
            assert result['branches']['uniform_off']['success']==4
            assert len(result['results'])==18*12
            source=next(r for r in result['results'] if r['geometry']=='beam' and r['region']=='full_field' and r['comparison']=='source')
            assert source['population_acceptance']=='withheld_numerical_nonresponse'
            exchange_uniform=next(r for r in result['results'] if r['geometry']=='beam' and r['region']=='full_field' and r['comparison']=='uniform_exchange')
            assert exchange_uniform['all_contributing_sampled_references_available']
    finally:
        c.ROOT,c.ready,pbs=previous
        if pbs is None:os.environ.pop('PBS_NCPUS',None)
        else:os.environ['PBS_NCPUS']=pbs


def main():
    population=np.array([1,3,100]);draws=c.allocation(population,17)
    assert draws.sum()==17 and np.all(draws<=population) and np.all(draws>0)
    assert draws[0]==1 and draws[1]==3
    y=np.array([1.,2.,5.,8.]);totals=[];variances=[]
    for pick in combinations(range(4),2):
        z=(2*y[list(pick)])[:,None]
        totals.append(z.sum());variances.append(joint_covariance(z,np.zeros(2,'i8'),np.array([4]),np.array([2]))[0,0])
    np.testing.assert_allclose(np.mean(totals),y.sum())
    np.testing.assert_allclose(np.mean(variances),np.var(totals))
    np.testing.assert_array_equal(joint_covariance(y[:,None],np.zeros(4,'i8'),np.array([4]),np.array([4])),0)
    # Finite-difference gradients for both complete-residual and matched-mask cases.
    sd=np.array([[3.,4.,2.],[6.,9.,3.]]);sp=sd*.9;exact=sp*10
    for complete in (False,True):
        def evaluate(vector):
            a,b=np.split(vector,2);a=a.reshape(2,3);b=b.reshape(2,3)
            d=exact+a-b if complete else a;p=exact if complete else b
            return log_metrics(d,p,complete,.01)
        vector=np.r_[sd.ravel(),sp.ravel()];metrics=evaluate(vector)
        gradient=metrics['direct_ratio_gradient']-metrics['proxy_ratio_gradient']
        for i in range(len(vector)):
            step=np.zeros_like(vector);step[i]=1e-5
            a,b=evaluate(vector+step),evaluate(vector-step)
            numeric=((a['direct_ratio']-a['proxy_ratio'])-(b['direct_ratio']-b['proxy_ratio']))/2e-5
            np.testing.assert_allclose(numeric,gradient[:,i],atol=1e-10,rtol=2e-7)
    rng=np.random.default_rng(7);field=rng.random((19,19));masks=rng.random((3,19,19));sigma=1.3;pixel=3.
    weights=adjoint(masks,sigma,pixel)
    beam=gaussian_filter(field,sigma,mode='constant')/gaussian_filter(np.ones(field.shape),sigma,mode='constant')*pixel**2
    np.testing.assert_allclose(np.sum(weights*field,axis=(1,2)),np.sum(masks*beam,axis=(1,2)),rtol=1e-13)
    ids=[np.arange(i,c.DRAW_COUNT*5,8) for i in range(8)]
    np.testing.assert_array_equal(np.sort(np.concatenate(ids)),np.arange(c.DRAW_COUNT*5))
    assert len(set(np.concatenate(ids)))==c.DRAW_COUNT*5
    # Failure is a matched-mask diagnostic, not proxy-filled population truth.
    proxy=np.ones((4,6,3));direct=proxy.copy();direct[0]=np.nan
    valid=np.ones((4,6),bool);valid[0]=False
    report=assess([0,1],direct,proxy,valid,np.ones(4),np.ones(4),np.zeros(4,'i8'),np.array([8]),np.array([4]),np.ones((6,3))*8,0.)
    assert not report['all_contributing_sampled_references_available']
    assert report['population_acceptance']=='withheld_numerical_nonresponse'
    np.testing.assert_allclose(report['direct_line_totals'],3)
    np.testing.assert_allclose(report['common_missing_predicted_line_fractions'],.25)
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'state.npz';c.atomic_npz(path,status=np.array([1,-1,0]),values=np.array([1,np.nan,np.nan]))
        with np.load(path,allow_pickle=False) as f:
            assert f['status'].tolist()==[1,-1,0] and np.isnan(f['values'][1:]).all()
    assembled_review()
    result=dict(passed=True,distinct_particle_design=True,capped_stratum_allocation=True,
        HT_total_enumeration=True,finite_population_variance_enumeration=True,census_variance_zero=True,
        paired_delta_gradients=True,Gaussian_adjoint_identity=True,
        disjoint_state_partition=True,nonresponse_not_imputed=True,atomic_numeric_storage=True,
        assembled_eight_partition_report=True,pair_specific_failure_masks=True,
        original_manifest_sha256=c.core.verify_code())
    c.save(c.ROOT/'local_checks.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
