"""PBS-only streaming x8 pilot inputs; never fit a model or read direct labels."""
import argparse
import json
import os
from pathlib import Path

import h5py
import joblib
import numpy as np
from scipy.ndimage import gaussian_filter

import process_snapshot563_lookup_v11 as prod
from workflow_config import load_config
from adaptive_local_surrogate import (AdaptiveLocalActualAbundanceSurrogate,
    OUTPUT_NAMES, raw_inputs_to_features, transform_outputs, inverse_transform_outputs)

ROOT = Path(__file__).resolve().parent
ISO = ROOT.parent
MODELS = ('production_full', 'production_slug', 'actual_on', 'actual_off', 'uniform_on', 'uniform_off')
REGIONS = ('full_field', 'inner_1_8', *[f'annulus_{a}_{a+1}' for a in range(1,8)])


def save(path, value):
    temporary = Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def verify():
    manifest = json.loads((ROOT/'manifest.json').read_text())
    for p, expected in manifest['hashes'].items():
        assert prod.file_sha256(Path(p)) == expected, p
    for entry in manifest['snapshots']:
        stat = Path(entry['path']).stat()
        assert (stat.st_size, stat.st_mtime_ns) == (entry['size'], entry['mtime_ns'])
    return prod.file_sha256(ROOT/'manifest.json')


def environment():
    sha = verify()
    settings = prod.load_settings(ISO/'highres_snapshot575_temporal/config.json')
    paths = prod.snapshot_files(settings)
    prod.validate_snapshot_header(paths,575)
    full = load_config(settings['chemistry']['source_workflow_config'])[0]
    slug = load_config(ISO/'highres_snapshot575_temporal/slug_only/source_workflow.json')[0]
    with np.load(settings['outputs']['context_npz']) as f:
        nh, oxygen = f['NH_map_cm2'].copy(), float(f['oxygen_scale'])
    context = json.loads((ISO/'isotope_fractionation_controls/x8_snapshot575/context.json').read_text())
    return sha, settings, paths, full, slug, nh, oxygen, context


def chunks(path, settings, full, slug, nh, oxygen):
    units = prod.unit_factors(settings)
    snap, proj = settings['snapshot'], settings['projection']
    cols, plan = prod.isotope_plan(full)
    scols, splan = prod.isotope_plan(slug)
    columns = sorted(set(cols+scols))
    xh = snap['hydrogen_mass_fraction']
    res = proj['column_resolution']; pixel = proj['map_size_pc']/res
    with h5py.File(path) as f:
        g = f['PartType0']
        for start in range(0,len(g['Masses']),100000):
            stop = min(start+100000,len(g['Masses'])); sl = slice(start,stop)
            xyz = g['Coordinates'][sl]*units['length_pc']
            h = g['SmoothingLength'][sl]*units['length_pc']
            mass = g['Masses'][sl]*units['mass_Msun']
            n = g['Density'][sl]*units['density_cgs']*xh/prod.MH_G
            dv = np.maximum(g['NormVelocityGradient'][sl]*units['dvdr_km_s_pc'],proj['minimum_dvdr_km_s_pc'])
            hii = g['GrackleHII'][sl]
            hii[g['InternalEnergy'][sl]>snap['hot_gas_internal_energy_threshold']] = xh
            metal = g['Metallicity'][sl,columns]
            iso = prod.isotope_abundances(metal,columns,plan,xh)
            siso = prod.isotope_abundances(metal,columns,splan,xh)
            pix = np.floor(xyz[:,:2]/pixel).astype('i8')+res//2
            footprint = np.all((pix>=0)&(pix<res),axis=1)
            col = np.zeros(len(mass)); col[footprint] = nh[pix[footprint,0],pix[footprint,1]]
            raw = np.column_stack((n,col,dv,iso['12C'],iso['13C'],iso['16O']*oxygen,iso['18O']*oxygen))
            sr = np.column_stack((n,col,dv,siso['12C'],siso['13C'],siso['16O']*oxygen,siso['18O']*oxygen))
            use = footprint & (hii<=snap['maximum_neutral_hii_fraction'])
            use &= np.isfinite(mass)&(mass>0)&np.isfinite(h)&(h>0)
            use &= np.all(np.isfinite(raw),axis=1)&np.all(raw[:,[0,1,2,3,5]]>0,axis=1)&np.all(raw[:,[4,6]]>=0,axis=1)
            if not use.any(): continue
            assert np.all(np.isfinite(sr[use])) and np.all(sr[use][:,[0,1,2,3,5]]>0) and np.all(sr[use][:,[4,6]]>=0)
            yield dict(actual=raw[use],slug=sr[use],mass=mass[use],xyz=xyz[use],h=h[use],
                row=np.arange(start,stop,dtype='i8')[use],particle_id=g['ParticleIDs'][sl][use])


def uniform(raw,ratios):
    result = raw.copy()
    for a,b,r in ((3,4,ratios[0]),(5,6,ratios[1])):
        total = raw[:,a]+raw[:,b]; result[:,b] = total/(1+r); result[:,a] = total-result[:,b]
        np.testing.assert_allclose(result[:,a]+result[:,b],total,rtol=1e-14)
    return result


def predictions(raw,slug,ratios,base,bundle):
    result, cover = [], []
    for r in (raw,slug):
        initial,diag = base.query(r,chunk_size=100000)
        result.append(np.column_stack([initial[n] for n in OUTPUT_NAMES])[:,3:])
        cover.append(diag['within_axis_bounds']&diag['within_joint_coverage'])
    for r in (raw,uniform(raw,ratios)):
        initial,diag = base.query(r,chunk_size=100000)
        bt = transform_outputs(r,np.column_stack([initial[n] for n in OUTPUT_NAMES]))
        features = raw_inputs_to_features(r); aug = np.column_stack((features,bt))
        for physics in ('on','off'):
            correction = np.column_stack([m.predict(aug) for m in bundle['models'][physics]])
            pred = inverse_transform_outputs(r,bt+correction,bundle['target_clips'][physics])
            bounds = bundle['feature_bounds'][physics]
            good = np.all((features>=bounds['min'])&(features<=bounds['max']),axis=1)
            result.append(pred[:,3:]); cover.append(good&diag['within_axis_bounds']&diag['within_joint_coverage'])
    values = np.stack(result,axis=1)
    assert np.all(np.isfinite(values)&(values>=0))
    return values,np.column_stack(cover)


def proposal(data,ratios):
    lum = data['lum_per_H']*data['nuclei'][:,None,None]
    raw = data['actual']; mass = data['mass']
    cr = raw[:,3]/np.maximum(raw[:,4],1e-30)
    ore = raw[:,5]/np.maximum(raw[:,6],1e-30)
    extreme = 1+np.abs(np.log10(cr/ratios[0]))+np.abs(np.log10(ore/ratios[1]))+(~np.all(data['covered'],axis=1))
    return np.column_stack((np.ones(len(mass)),mass,lum[:,:,1].max(axis=1),lum[:,:,2].max(axis=1),mass*extreme))


def stratify(data):
    radius = np.hypot(*data['xyz'][:,:2].T)/1000
    return np.digitize(radius,np.arange(1,9))*3+np.digitize(np.log10(data['actual'][:,0]),[1,3])


def region_masks(xyz):
    r = np.hypot(*xyz[:,:2].T)/1000
    return np.column_stack((np.ones(len(r),bool),(r>=1)&(r<8),*[(r>=a)&(r<a+1) for a in range(1,8)]))


def append(group,name,value):
    if name not in group:
        group.create_dataset(name,shape=(0,*value.shape[1:]),maxshape=(None,*value.shape[1:]),dtype=value.dtype,chunks=True)
    d = group[name]; old = len(d); d.resize(old+len(value),axis=0); d[old:] = value


def cache(part):
    sha,settings,paths,full,slug,nh,oxygen,context = environment()
    target = ROOT/f'cache/part{part:02d}.h5'
    assert not target.exists(), 'Never overwrite an existing cache'
    base = AdaptiveLocalActualAbundanceSurrogate(settings['chemistry']['surrogate_model'],require_validated=False)
    frozen = json.loads((ISO/'isotope_fractionation_controls/config_final600.json').read_text())
    bundle = joblib.load(frozen['frozen_model'])
    assert prod.file_sha256(Path(bundle['base_model'])) == bundle['base_sha256']
    assert Path(bundle['base_model']).resolve() == Path(settings['chemistry']['surrogate_model']).resolve()
    component_sums = np.zeros((27,5)); totals = np.zeros((9,6,3)); coverage = np.zeros((6,4))
    count = 0; mass_sum = 0.
    with h5py.File(target,'x') as f:
        f.attrs.update(complete=False,manifest_sha256=sha,part=part,oxygen_scale=oxygen)
        for data in chunks(paths[part],settings,full,slug,nh,oxygen):
            data['nuclei'] = data['mass']*prod.MSUN_G*settings['snapshot']['hydrogen_mass_fraction']/prod.MH_G
            data['lum_per_H'],data['covered'] = predictions(data['actual'],data['slug'],context['ratios'],base,bundle)
            data['stratum'] = stratify(data)
            weights = proposal(data,context['ratios'])
            np.add.at(component_sums,data['stratum'],weights)
            lum = data['nuclei'][:,None,None]*data['lum_per_H']
            for j,mask in enumerate(region_masks(data['xyz']).T): totals[j] += lum[mask].sum(axis=0)
            for j in range(6):
                bad = ~data['covered'][:,j]
                coverage[j] += np.r_[data['mass'][bad].sum(),lum[bad,j].sum(axis=0)]
            for k,v in data.items(): append(f,k,v)
            count += len(data['mass']); mass_sum += data['mass'].sum()
            print(f'part={part}, active={count}',flush=True)
        for k,v in dict(component_sums=component_sums,predicted_particle_totals=totals,outside_coverage=coverage).items(): f.create_dataset(k,data=v)
        f.attrs.update(active_count=count,active_mass_Msun=mass_sum,complete=True)


def map_masks(settings):
    size = settings['projection']['map_size_pc']; res = settings['projection']['map_resolution']
    xy = (np.arange(res)+.5)*size/res-size/2
    x,y = np.meshgrid(xy,xy,indexing='ij')
    return region_masks(np.column_stack((x.ravel(),y.ravel()))).T.reshape((9,res,res))


def smooth(field,sigma):
    # Match the finite-Gaussian normalization in the source-comparison pipeline.
    assert np.all(np.isfinite(field))
    numerator=gaussian_filter(field,sigma,mode='constant',cval=0)
    denominator=gaussian_filter(np.ones(field.shape),sigma,mode='constant',cval=0)
    return numerator/denominator


def kernel(data,settings):
    from scipy.ndimage import gaussian_filter
    proj = settings['projection']; size = proj['map_size_pc']; res = proj['map_resolution']; pixel = size/res
    masks = map_masks(settings); sigma = 125/(np.sqrt(8*np.log(2))*pixel)
    result = []
    for i in range(len(data['mass'])):
        unit = prod.project_surface(data['xyz'][i:i+1],data['h'][i:i+1],np.ones(1),size,res)
        beam = smooth(unit,sigma)*pixel**2
        assert np.all(np.isfinite(beam)) and beam.sum()>0
        result.append([beam[m].sum() for m in masks])
    return np.asarray(result)


def exact_maps(settings):
    masks = map_masks(settings); pixel = settings['projection']['map_size_pc']/settings['projection']['map_resolution']
    sigma = 125/(np.sqrt(8*np.log(2))*pixel); arrays = []
    for folder in ('output','slug_only/output'):
        with np.load(ISO/f'highres_snapshot575_temporal/{folder}/snapshot_575_maps_chuhan.npz') as f:
            arrays.append(np.stack([f['mapL'+s] for s in ('CO','13CO','C18O')]))
    with np.load(ISO/'isotope_fractionation_controls/x8_snapshot575/maps_native.npz') as f:
        arrays.extend([f[b][3:] for b in MODELS[2:]])
    totals = np.zeros((9,6,3))
    for b,arr in enumerate(arrays):
        for j,field in enumerate(arr):
            beam = smooth(field,sigma)*pixel**2
            totals[:,b,j] = [beam[m].sum() for m in masks]
    return totals


def sample():
    sha,settings,_,_,_,_,_,context = environment()
    target = ROOT/'pilot_sample.h5'; assert not target.exists()
    paths = [ROOT/f'cache/part{i:02d}.h5' for i in range(8)]
    sums = np.zeros((27,5)); predicted = np.zeros((9,6,3)); outside = np.zeros((6,4))
    count = 0; mass = 0.
    for p in paths:
        with h5py.File(p) as f:
            assert f.attrs['complete'] and f.attrs['manifest_sha256']==sha
            sums += f['component_sums'][:]; predicted += f['predicted_particle_totals'][:]; outside += f['outside_coverage'][:]
            count += f.attrs['active_count']; mass += f.attrs['active_mass_Msun']
    assert count==context['active_count']
    assert np.isclose(mass,context['active_mass_Msun'],rtol=1e-10)
    # Compare independent streaming sums against the existing Full/SLUG and paired runs.
    fullsummary = json.loads((ISO/'highres_snapshot575_temporal/output/snapshot575_summary.json').read_text())
    slugsummary = json.loads((ISO/'highres_snapshot575_temporal/slug_only/output/snapshot575_summary.json').read_text())
    closure = []
    for b,summary in enumerate((fullsummary,slugsummary)):
        old = summary['total_line_luminosity_erg_s']
        old = np.asarray([old[n] for n in ('CO','13CO','C18O')]) if isinstance(old,dict) else np.asarray(old)
        delta = np.log10(predicted[0,b]/old); assert np.max(np.abs(delta))<1e-6,delta
        closure.append(delta.tolist())
    for b,name in enumerate(MODELS[2:],2):
        old = np.zeros(3)
        for part in range(8):
            with np.load(ISO/f'isotope_fractionation_controls/x8_snapshot575/part{part:02d}.npz') as f: old += f[name+'_totals'][2:5]
        delta = np.log10(predicted[0,b]/old); assert np.max(np.abs(delta))<1e-6,delta
        closure.append(delta.tolist())
    groups = np.flatnonzero(sums[:,0]>0)
    m = np.zeros(27,dtype='i8'); m[groups] = 128//len(groups); m[groups[:128%len(groups)]] += 1
    assert m[groups].min()>=4
    rng = np.random.default_rng(20261027575)
    targets = {int(h):np.sort(rng.random(m[h])) for h in groups}
    cumulative = np.zeros(27); cursor = np.zeros(27,dtype='i8'); rows = []
    mixture = np.array([.2,.2,.25,.25,.1])
    fields = ('actual','slug','mass','xyz','h','row','particle_id','nuclei','lum_per_H','covered','stratum')
    for part,p in enumerate(paths):
        with h5py.File(p) as f:
            for start in range(0,len(f['mass']),100000):
                data = {k:f[k][start:start+100000] for k in fields}
                w = proposal(data,context['ratios'])
                for h in np.unique(data['stratum']):
                    idx = np.flatnonzero(data['stratum']==h)
                    normalized = np.empty((len(idx),5))
                    for c in range(5): normalized[:,c] = w[idx,c]/sums[h,c] if sums[h,c]>0 else 1/sums[h,0]
                    q = normalized@mixture
                    cdf = cumulative[h]+np.cumsum(q)
                    last = np.searchsorted(targets[int(h)],cdf[-1],side='left')
                    for t in targets[int(h)][cursor[h]:last]:
                        j = int(np.searchsorted(cdf,t,side='right')); local = int(idx[j])
                        rows.append({k:data[k][local] for k in fields}|dict(part=part,q=q[j],draws_in_stratum=m[h]))
                    cursor[h] = last; cumulative[h] = cdf[-1]
    np.testing.assert_allclose(cumulative[groups],1,rtol=1e-10,atol=1e-12)
    assert np.array_equal(cursor,m) and len(rows)==128
    data = {k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    identity = np.column_stack((data['part'],data['row']))
    unique,first,inverse = np.unique(identity,axis=0,return_index=True,return_inverse=True)
    selected = {k:v[first] for k,v in data.items()}
    selected['uniform'] = uniform(selected['actual'],context['ratios'])
    selected['kernel_weights'] = kernel(selected,settings)
    # SurfaceDensity is an additive kernel: test single-particle sum against a joint projection.
    test = {k:v[:min(4,len(first))] for k,v in selected.items()}
    proj = settings['projection']; size=proj['map_size_pc']; res=proj['map_resolution']; pixel=size/res
    joint = prod.project_surface(test['xyz'],test['h'],np.ones(len(test['mass'])),size,res)
    joint = smooth(joint,125/(np.sqrt(8*np.log(2))*pixel))*pixel**2
    got = np.asarray([joint[mask].sum() for mask in map_masks(settings)])
    np.testing.assert_allclose(got,test['kernel_weights'].sum(axis=0),rtol=2e-5,atol=1e-8)
    with h5py.File(target,'x') as f:
        f.attrs.update(complete=False,manifest_sha256=sha,models_json=json.dumps(MODELS),regions_json=json.dumps(REGIONS))
        for k,v in selected.items(): f.create_dataset('unique/'+k,data=v)
        for k in ('q','draws_in_stratum','stratum'): f.create_dataset('draw/'+k,data=data[k])
        f.create_dataset('draw/unique_index',data=inverse)
        f.create_dataset('draw/HH_factor',data=1/(data['q']*data['draws_in_stratum']))
        f.create_dataset('unique/inclusion_probability',data=-np.expm1(selected['draws_in_stratum']*np.log1p(-selected['q'])))
        f.create_dataset('predicted_particle_totals',data=predicted)
        f.create_dataset('predicted_beam_totals',data=exact_maps(settings))
        f.create_dataset('outside_coverage',data=outside)
        f.attrs.update(complete=True,active_count=count,active_mass_Msun=mass)
    save(ROOT/'preparation_review.json',dict(complete=True,manifest_sha256=sha,draws=128,unique_particles=len(unique),
        physical_states=5*len(unique),active_count=int(count),active_mass_Msun=float(mass),
        stratum_counts=m.tolist(),sum_draw_probabilities=cumulative.tolist(),
        existing_prediction_closure_dex=closure,kernel_additivity_passed=True,
        warning='128 importance draws are a pilot, not guaranteed annular precision.'))
    print(f'Saved 128 draws, {len(unique)} particles, {5*len(unique)} physical states',flush=True)


if __name__=='__main__':
    assert int(os.environ.get('PBS_NCPUS','0'))>0, 'Preparation requires PBS'
    p = argparse.ArgumentParser(); p.add_argument('stage',choices=('cache','sample')); p.add_argument('--part',type=int,default=0)
    a=p.parse_args()
    if a.stage=='cache': cache(a.part)
    else: sample()
