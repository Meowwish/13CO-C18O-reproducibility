"""Paired spatial-claim checks on existing direct labels; no new chemistry."""
import json
import os

import h5py
import numpy as np
from scipy.stats import norm

import claim_common as c


def main():
    assert int(os.environ.get('PBS_NCPUS','0'))>0
    sha,_=c.ready();assert not (c.ROOT/'review.json').exists()
    with h5py.File(c.SOURCE/'sample.h5') as f:
        n=len(f['sample/mass']);mass=f['sample/mass'][:];nuclei=f['sample/nuclei'][:]
        pi=f['sample/inclusion_probability'][:];strata=f['sample/stratum'][:];population=f['stratum_population'][:]
    design=c.sfr.groups(strata,population)
    with np.load(c.ROOT/'predictions.npz') as f:pred=f['predicted']
    kernel=np.full((n,2,len(c.REGIONS)),np.nan);seen=np.zeros(n,bool)
    for part in range(4):
        with np.load(c.ROOT/f'geometry/part{part:02d}.npz') as f:
            assert bool(f['complete']) and str(f['manifest_sha256'])==sha
            ids=f['ids'];assert not seen[ids].any();kernel[ids]=f['kernel'];seen[ids]=True
    assert seen.all() and np.all(np.isfinite(kernel))
    sfr_kernel=np.full((n,80),np.nan);seen=np.zeros(n,bool)
    for part in range(4):
        with np.load(c.SFR/f'geometry/part{part:02d}.npz') as f:
            ids=f['ids'];assert not seen[ids].any();sfr_kernel[ids]=f['kernel'];seen[ids]=True
    assert seen.all() and np.all(np.isfinite(sfr_kernel))
    catalogue=json.loads((c.ROOT/'catalogue.json').read_text())
    records=[];shapes=[];slopes=[];sfr_pairs=[]
    for reference,path in c.REFERENCES:
        with np.load(path) as f:values,qualified=f['values'],f['qualified']
        for model,name in enumerate(c.MODELS):
            q=qualified[:,c.MODEL_BRANCH[model]];k=kernel[:,int(model>=2)]
            molecular=c.estimate(values[:,c.MODEL_BRANCH[model],4:6],pred[:,model,1:3],q,k,mass/pi)
            line=c.estimate(values[:,c.MODEL_BRANCH[model],7:9],pred[:,model,4:6],q,k,nuclei/pi)
            effective={key:line[key]-molecular[key] for key in ('direct','proxy','direct_score','proxy_score')}
            for observable,m in [('line_ratio',line),('molecule_number_ratio',molecular),('emission_per_molecule_ratio',effective)]:
                for j in range(1,8):
                    sd=m['direct_score'][:,j];sp=m['proxy_score'][:,j]
                    error=c.sfr.interval(m['direct'][j]-m['proxy'][j],sd-sp,design)
                    ci=error['interval_95']
                    records.append(dict(reference=reference,model=name,observable=observable,region=c.REGIONS[j],
                        direct_log_ratio=c.sfr.interval(m['direct'][j],sd,design),
                        matched_surrogate_log_ratio=m['proxy'][j],direct_minus_surrogate=error,
                        interval_within_0p05_dex=ci is not None and ci[0]>=-.05 and ci[1]<=.05,
                        missing_predicted_line_fraction=line['missing_predicted_positive_fraction'][j],
                        missing_predicted_molecular_fraction=molecular['missing_predicted_positive_fraction'][j]))
                for j in range(1,7):
                    a=np.zeros(len(c.REGIONS));a[j:j+2]=[-1,1]
                    result,sd,sp=c.contrast(m,a,design)
                    z=float(norm.isf(.05/12));se=result['direct']['standard_error'];point=result['direct']['point']
                    shapes.append(dict(reference=reference,model=name,observable=observable,
                        contrast=f'annulus_{j+1}_{j+2}_minus_{j}_{j+1}',**result,
                        within_inner_curve_familywise_95=[point-z*se,point+z*se] if se is not None else None))
        # All four Full-source controls use the identical common-success mask.
        common=qualified[:,[0,2,3,4]].all(axis=1)
        cached={}
        for model in range(2,6):
            name=c.MODELS[model]
            m=c.estimate(values[:,c.MODEL_BRANCH[model],7:9],pred[:,model,4:6],common,sfr_kernel,nuclei/pi)
            for record in catalogue['coefficients']:
                a=np.array(record['coefficient']);result,sd,sp=c.contrast(m,a,design)
                key=(record['window_Myr'],str(record['radial']),model)
                cached[key]=(result,sd,sp)
                slopes.append(dict(reference=reference,model=name,window_Myr=record['window_Myr'],radial=record['radial'],
                    reference_mask='four_control_common_success',**result,
                    maximum_missing_predicted_line_fraction=np.nanmax(m['missing_predicted_positive_fraction'][a!=0])))
        for record in catalogue['coefficients']:
            for name,(one,two) in dict(exchange=(2,3),uniform_exchange=(4,5),pattern_on=(2,4),pattern_off=(3,5)).items():
                first=cached[record['window_Myr'],str(record['radial']),one]
                second=cached[record['window_Myr'],str(record['radial']),two]
                d=first[0]['direct']['point']-second[0]['direct']['point']
                p=first[0]['matched_surrogate']-second[0]['matched_surrogate']
                sd=first[1]-second[1];sp=first[2]-second[2]
                sfr_pairs.append(dict(reference=reference,comparison=name,window_Myr=record['window_Myr'],radial=record['radial'],
                    direct_slope_difference=c.sfr.interval(d,sd,design),matched_surrogate_difference=p,
                    difference_error=c.sfr.interval(d-p,sd-sp,design)))
        print('Reviewed claim reference',reference,flush=True)
    c.save(c.ROOT/'review.json',dict(complete=True,manifest_sha256=sha,
        model_training_performed=False,new_chemical_solves=0,radial_records=records,inner_adjacent_contrasts=shapes,
        control_sfr_slopes=slopes,paired_sfr_slope_differences=sfr_pairs,
        scientific_acceptance=False,scope='Conditional successful-state, probability-weighted ratios and correlated contrasts; no physical-error or full-population bound.'))


if __name__=='__main__':main()
