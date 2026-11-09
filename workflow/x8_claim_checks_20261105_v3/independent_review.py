"""Independent cached sums and scalar variance checks, separate from estimators."""
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import norm

ROOT=Path(__file__).resolve().parent
ISO=ROOT.parent
SOURCE=ISO/'x8_bounded_reference_20261101'
MODELS=('production_full','production_slug','actual_on','actual_off','uniform_on','uniform_off')
BRANCH=[0,1,0,2,3,4]


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    assert int(os.environ.get('PBS_NCPUS','0'))>0
    r=json.loads((ROOT/'review.json').read_text())
    assert r['complete'] and r['manifest_sha256']==digest(ROOT/'manifest.json')
    for path,sha in json.loads((ROOT/'manifest.json').read_text())['hashes'].items():assert digest(path)==sha,path
    with h5py.File(SOURCE/'sample.h5') as f:
        mass,nuclei,pi,strata=[f['sample/'+name][:] for name in ('mass','nuclei','inclusion_probability','stratum')]
        pop,allocation=f['stratum_population'][:],f['stratum_sample'][:]
    np.testing.assert_array_equal(pi,allocation[strata]/pop[strata])
    np.testing.assert_array_equal(np.bincount(strata,minlength=len(pop)),allocation)
    n=len(pi);groups=[(np.flatnonzero(strata==h),pop[h]) for h in np.unique(strata)]
    with np.load(ROOT/'predictions.npz') as f:pred=f['predicted']
    kernel=np.full((n,2,19),np.nan);sfr=np.full((n,80),np.nan)
    output_hashes={}
    for folder,key,target in ((ROOT,'kernel',kernel),(ISO/'x8_sfr_validation_20261102','kernel',sfr)):
        seen=np.zeros(n,bool)
        for part in range(4):
            path=folder/f'geometry/part{part:02d}.npz';output_hashes[str(path)]=digest(path)
            with np.load(path) as f:
                ids=f['ids'];assert not seen[ids].any() and bool(f['complete'])
                target[ids]=f[key];seen[ids]=True
        assert seen.all() and np.all(np.isfinite(target))
    bins=json.loads((ROOT/'catalogue.json').read_text())['bins']
    def coefficients(window,radial):
        # Reconstruct the common coefficient by demeaning each eligible
        # annulus independently, rather than importing the production helper.
        a=np.zeros(80)
        radial_groups=range(1,8) if radial=='fixed_effect' else [0]
        denominator=0.
        for b in radial_groups:
            ids=[i for i,x in enumerate(bins) if x['eligible'] and x['window_Myr']==window and x['radial_bin']==b]
            if len(ids)<(2 if radial=='fixed_effect' else 3):continue
            x=np.array([(bins[i]['log_sfr_lo']+bins[i]['log_sfr_hi'])/2 for i in ids])
            dx=x-x.mean();a[ids]=dx;denominator+=np.sum(dx*dx)
        assert denominator>0
        return a/denominator
    def pair(values,prediction,mask,k,w):
        v=np.where(mask[:,None],values,0.);p=np.where(mask[:,None],prediction,0.)
        direct=np.zeros(k.shape[1]);proxy=direct.copy();sd=np.zeros_like(k);sp=sd.copy()
        for j in range(k.shape[1]):
            a=w*k[:,j];dv=np.sum(a[:,None]*v,axis=0);pv=np.sum(a[:,None]*p,axis=0)
            assert np.all(dv>0) and np.all(pv>0)
            direct[j]=np.log10(dv[0])-np.log10(dv[1]);proxy[j]=np.log10(pv[0])-np.log10(pv[1])
            sd[:,j]=a*(v[:,0]/dv[0]-v[:,1]/dv[1])/np.log(10)
            sp[:,j]=a*(p[:,0]/pv[0]-p[:,1]/pv[1])/np.log(10)
        return direct,proxy,sd,sp
    checked=0;point_difference=0.;se_difference=0.
    def check(record,value,score):
        nonlocal checked,point_difference,se_difference
        assert record['point'] is not None and record['standard_error'] is not None
        variance=0.
        for ids,N in groups:
            m=len(ids)
            if m<N:
                centred=score[ids]-np.mean(score[ids])
                variance+=m*(1-m/N)*np.dot(centred,centred)/(m-1)
        se=float(np.sqrt(max(variance,0.)))
        point_difference=max(point_difference,abs(record['point']-value))
        se_difference=max(se_difference,abs(record['standard_error']-se))
        np.testing.assert_allclose(record['point'],value,rtol=0,atol=3e-12)
        np.testing.assert_allclose(record['standard_error'],se,rtol=0,atol=3e-12)
        np.testing.assert_allclose(record['interval_95'],[value-1.96*se,value+1.96*se],rtol=0,atol=1e-11)
        checked+=1
    for label,path in [('default',ISO/'x8_reference_recovery_20261101/candidate_overlay.npz'),
                       ('extended',ISO/'x8_failure_diagnostics_20261102/diagnostic_candidates.npz')]:
        with np.load(path) as f:values,qualified=f['values'],f['qualified']
        for model,name in enumerate(MODELS):
            branch=BRANCH[model];q=qualified[:,branch];k=kernel[:,int(model>=2)]
            # Inner regions are the validated targets. Empty exterior
            # regions are irrelevant to this independent calculation.
            k=k[:,1:8]
            line=pair(values[:,branch,7:9],pred[:,model,4:6],q,k,nuclei/pi)
            molecule=pair(values[:,branch,4:6],pred[:,model,1:3],q,k,mass/pi)
            for observable,m in [('line_ratio',line),('molecule_number_ratio',molecule),
                    ('emission_per_molecule_ratio',tuple(a-b for a,b in zip(line,molecule)))]:
                for row in r['radial_records']:
                    if (row['reference'],row['model'],row['observable'])!=(label,name,observable):continue
                    j=int(row['region'].split('_')[1])-1
                    check(row['direct_log_ratio'],m[0][j],m[2][:,j])
                    check(row['direct_minus_surrogate'],m[0][j]-m[1][j],m[2][:,j]-m[3][:,j])
                for row in r['inner_adjacent_contrasts']:
                    if (row['reference'],row['model'],row['observable'])!=(label,name,observable):continue
                    j=int(row['contrast'].split('_')[1])-2
                    d=m[0][j+1]-m[0][j];p=m[1][j+1]-m[1][j]
                    sd=m[2][:,j+1]-m[2][:,j];sp=m[3][:,j+1]-m[3][:,j]
                    check(row['direct'],d,sd);check(row['direct_minus_surrogate'],d-p,sd-sp)
                    z=norm.isf(.05/12);se=row['direct']['standard_error']
                    np.testing.assert_allclose(row['within_inner_curve_familywise_95'],[d-z*se,d+z*se],atol=1e-11,rtol=0)
        common=qualified[:,[0,2,3,4]].all(axis=1);cached={}
        for model in range(2,6):
            name=MODELS[model];branch=BRANCH[model]
            # Keep only eligible cells, avoiding unrelated empty bins.
            use=np.array([row['eligible'] for row in bins]);m=pair(values[:,branch,7:9],pred[:,model,4:6],common,sfr[:,use],nuclei/pi)
            for row in r['control_sfr_slopes']:
                if (row['reference'],row['model'])!=(label,name):continue
                a=coefficients(row['window_Myr'],row['radial'])[use]
                d=a@m[0];p=a@m[1];sd=m[2]@a;sp=m[3]@a
                check(row['direct'],d,sd);check(row['direct_minus_surrogate'],d-p,sd-sp)
                cached[row['window_Myr'],str(row['radial']),model]=(d,p,sd,sp)
        pairs=dict(exchange=(2,3),uniform_exchange=(4,5),pattern_on=(2,4),pattern_off=(3,5))
        for row in r['paired_sfr_slope_differences']:
            if row['reference']!=label:continue
            one,two=pairs[row['comparison']]
            first=cached[row['window_Myr'],str(row['radial']),one];second=cached[row['window_Myr'],str(row['radial']),two]
            d,p,sd,sp=[a-b for a,b in zip(first,second)]
            check(row['direct_slope_difference'],d,sd);check(row['difference_error'],d-p,sd-sp)
    report=dict(passed=True,manifest_sha256=r['manifest_sha256'],review_sha256=digest(ROOT/'review.json'),
        independently_checked_intervals=checked,maximum_point_difference=point_difference,
        maximum_standard_error_difference=se_difference,geometry_hashes=output_hashes,
        scope='Independent scalar influence reconstruction; no physical or full-population error bound.')
    (ROOT/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Independent checks passed:',checked,'intervals.',flush=True)


if __name__=='__main__':main()
