"""Replay frozen molecular predictions and construct identical control SFR bins."""
import json
import os

import h5py
import joblib
import numpy as np

import claim_common as c
from adaptive_local_surrogate import (AdaptiveLocalActualAbundanceSurrogate, OUTPUT_NAMES,
    transform_outputs, raw_inputs_to_features, inverse_transform_outputs)
from plot_galic_selection import lattice, sample


def main():
    assert int(os.environ.get('PBS_NCPUS','0'))>0
    sha = c.verify()
    assert not (c.ROOT/'prepared.json').exists()
    with h5py.File(c.SOURCE/'sample.h5') as f:
        actual,slug,uniform = [f['sample/'+key][:] for key in ('actual','slug','uniform')]
        previous = f['sample/lum_per_H'][:]
    model_path = c.ISO/'chemTable_temporal_550_575/final_test_600/frozen_surrogate.joblib'
    base = AdaptiveLocalActualAbundanceSurrogate(model_path,require_validated=False)
    bundle = joblib.load(c.ISO/'isotope_fractionation_controls/intervention_models/residual_leaves15.joblib')
    assert c.sha(model_path)==bundle['base_sha256']
    predictions = []
    for raw in (actual,slug):
        values,_ = base.query(raw,chunk_size=100000)
        predictions.append(np.column_stack([values[name] for name in OUTPUT_NAMES]))
    for raw in (actual,uniform):
        values,_ = base.query(raw,chunk_size=100000)
        bt = transform_outputs(raw,np.column_stack([values[name] for name in OUTPUT_NAMES]))
        aug = np.column_stack((raw_inputs_to_features(raw),bt))
        for physics in ('on','off'):
            correction = np.column_stack([model.predict(aug) for model in bundle['models'][physics]])
            predictions.append(inverse_transform_outputs(raw,bt+correction,bundle['target_clips'][physics]))
    predictions = np.stack(predictions,axis=1)
    np.testing.assert_allclose(predictions[:,:,3:],previous,rtol=2e-12,atol=1e-60)
    assert np.all(np.isfinite(predictions)) and np.all(predictions>=0)
    c.base.atomic_npz(c.ROOT/'predictions.npz',predicted=predictions,manifest_sha256=np.asarray(sha))
    # Actual maps are sampled on exactly the previous catalogue's common
    # footprint; all four controls use the same SFR field and bin coefficients.
    xy = lattice(); radius=np.hypot(*xy.T)
    fields=[]
    for path in (c.ISO/'highres_snapshot575_temporal/output/snapshot_575_maps_125pc.npz',
                 c.ISO/'highres_snapshot575_temporal/slug_only/output/snapshot_575_maps_125pc.npz'):
        with np.load(path) as f:
            fields.append(np.column_stack([sample(f[k],xy) for k in ('W13CO','WC18O')]))
    with np.load(c.ISO/'isotope_fractionation_controls/x8_snapshot575/maps_125pc.npz') as f:
        for name in c.MODELS[2:]:
            fields.append(np.column_stack([sample(f[name+'_'+k],xy) for k in ('W13CO','WC18O')]))
    fields=np.stack(fields,axis=1)
    assert np.all(np.isfinite(fields)) and np.all(fields>=0)
    catalogue=json.loads((c.SFR/'catalogue.json').read_text())['bins']
    sectors=np.floor((np.arctan2(xy[:,1],xy[:,0])+np.pi)/(2*np.pi)*12).astype(int)%12
    totals=np.zeros((80,6,2)); leave=np.zeros((12,80,6,2))
    with np.load(c.sfr.SFR) as f:
        windows={w:sample(f[f'{w}Myr_125pc'],xy) for w in (1,10)}
    for i,meta in enumerate(catalogue):
        sfr=windows[meta['window_Myr']];r=meta['radial_bin']
        mask=(radius>=1)&(radius<8)&np.isfinite(sfr)&(sfr>0)
        if r:mask&=(radius>=r)&(radius<r+1)
        mask&=(sfr>=10**meta['log_sfr_lo'])&(sfr<10**meta['log_sfr_hi'])
        assert mask.sum()==meta['count'],(i,mask.sum(),meta['count'])
        totals[i]=fields[mask].sum(0)
        for sector in range(12):leave[sector,i]=fields[mask&(sectors!=sector)].sum(0)
        for model,key in ((0,'Full'),(1,'SLUG')):
            if meta['eligible']:
                np.testing.assert_allclose(totals[i,model,0]/totals[i,model,1],meta['known_population_ratio'][key],rtol=2e-12)
    slopes=[];paired=[]
    coefficient_records=[]
    for window in (1,10):
        for radial in (0,'fixed_effect'):
            a=c.sfr.coefficients(catalogue,window,radial)
            assert a is not None
            use=a!=0
            assert np.all(totals[use]>0)
            y=np.log10(totals[use,:,0]/totals[use,:,1])
            point=a[use]@y
            replicas = None
            errors = [None]*len(c.MODELS)
            if np.all(leave[:,use]>0):
                replicas=c.spatial_slope_replicas(a,leave)
                errors=np.sqrt(11/12*np.sum((replicas-replicas.mean(0))**2,axis=0))
            for model,name in enumerate(c.MODELS):
                slopes.append(dict(window_Myr=window,radial=radial,model=name,slope=point[model],spatial_jackknife=errors[model]))
            for name,(one,two) in dict(exchange=(2,3),uniform_exchange=(4,5),pattern_on=(2,4),pattern_off=(3,5)).items():
                delta=replicas[:,one]-replicas[:,two] if replicas is not None else None
                paired.append(dict(window_Myr=window,radial=radial,comparison=name,slope_difference=point[one]-point[two],
                    paired_spatial_jackknife=np.sqrt(11/12*np.sum((delta-delta.mean())**2)) if delta is not None else None))
            coefficient_records.append(dict(window_Myr=window,radial=radial,coefficient=a))
    c.save(c.ROOT/'map_sfr.json',dict(slopes=slopes,paired_slope_differences=paired,
        fixed_eligible_bins=True,sector_count=12,common_footprint=True,scope='Frozen surrogate maps; spatial jackknife is not physical-model or reference uncertainty.'))
    c.save(c.ROOT/'catalogue.json',dict(bins=catalogue,coefficients=coefficient_records))
    c.save(c.ROOT/'prepared.json',dict(complete=True,manifest_sha256=sha,
        hashes={str(c.ROOT/name):c.sha(c.ROOT/name) for name in ('predictions.npz','map_sfr.json','catalogue.json')},
        sample_particles=len(actual),new_chemical_solves=0,luminosity_replay_passed=True))
    print('Prepared molecular replay and common four-control SFR statistics.',flush=True)


if __name__=='__main__':main()
