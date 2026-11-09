"""PBS-only 1 Myr frozen-surrogate sequence, common Full oxygen calibration per epoch."""
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import multiprocessing
import os
import time
import h5py
import joblib
import numpy as np
from scipy.ndimage import gaussian_filter
import extension as c
import process_snapshot563_lookup_v11 as prod
from workflow_config import load_config
from adaptive_local_surrogate import AdaptiveLocalActualAbundanceSurrogate

pilot = c.module('extension_pilot_reader',c.ISO/'x8_direct_validation_20261027/prepare.py')
THREAD_LIMIT = None


def initialize():
    global THREAD_LIMIT
    from threadpoolctl import threadpool_limits
    import numba
    numba.set_num_threads(13)
    THREAD_LIMIT = threadpool_limits(limits=13, user_api='openmp')


def environment(snap):
    root = c.ROOT/f'time/{snap}'
    settings = prod.load_settings(root/'config.json')
    paths = prod.snapshot_files(settings)
    full = load_config(settings['chemistry']['source_workflow_config'])[0]
    slug = load_config(c.ISO/'highres_snapshot575_temporal/slug_only/source_workflow.json')[0]
    with np.load(root/'context.npz') as f:
        nh, oxygen = f['NH_map_cm2'].copy(), float(f['oxygen_scale'])
    return root,settings,paths,full,slug,nh,oxygen


def reservoir(task):
    snap, part = task
    root,settings,paths,full,slug,nh,oxygen = environment(snap)
    target = root/f'reservoir{part:02d}.npz'
    if target.exists():
        with np.load(target) as f: return f['sums'].copy()
    sums = np.zeros(6)
    for data in pilot.chunks(paths[part],settings,full,slug,nh,oxygen):
        sums[:4] += (data['mass'][:,None]*data['actual'][:,3:7]).sum(axis=0)
        sums[4] += data['mass'].sum(); sums[5] += len(data['mass'])
    c.design.atomic_npz(target,sums=sums)
    print(snap,part,'reservoir done',flush=True)
    return sums


def stellar_maps(path, settings):
    projection = settings['projection']; res = projection['map_resolution']; size = projection['map_size_pc']
    edges = np.linspace(-size/2,size/2,res+1); units = prod.unit_factors(settings)
    maps = np.zeros((2,res,res)); audit = np.zeros((2,4),dtype='i8')
    with h5py.File(path) as f:
        if 'PartType4' not in f: return maps,audit
        p = f['PartType4']; now = float(f['Header'].attrs['Time'])
        for start in range(0,len(p['Coordinates']),500000):
            sl = slice(start,start+500000)
            xy = p['Coordinates'][sl,:2]*units['length_pc']
            age = (now-p['StellarFormationTime'][sl])*1000
            mass = p['SlugStateDouble'][sl,0]
            footprint = np.all((xy>=-size/2)&(xy<size/2),axis=1)&np.all(np.isfinite(xy),axis=1)
            for k,window in enumerate((1,10)):
                use = footprint&np.isfinite(age)&(age>=0)&(age<window)
                valid = use&np.isfinite(mass)&(mass>0)
                audit[k] += [int(use.sum()),int(valid.sum()),int((use&~valid).sum()),
                             int((footprint&~np.isfinite(age)).sum())]
                maps[k] += np.histogram2d(xy[valid,0],xy[valid,1],bins=(edges,edges),
                             weights=mass[valid]/(window*1e6*(size/res/1000)**2))[0]
    return maps,audit


def predict_part(task):
    snap, part, ratios, sha = task
    root,settings,paths,full,slug,nh,oxygen = environment(snap)
    target = root/f'part{part:02d}.npz'
    if target.exists():
        with np.load(target) as f: assert str(f['manifest_sha256']) == sha
        return str(target)
    base = AdaptiveLocalActualAbundanceSurrogate(settings['chemistry']['surrogate_model'],require_validated=False)
    control = json.loads((c.ISO/'isotope_fractionation_controls/config_final600.json').read_text())
    bundle = joblib.load(control['frozen_model'])
    assert c.digest(bundle['base_model']) == bundle['base_sha256']
    assert c.Path(bundle['base_model']).resolve() == c.Path(settings['chemistry']['surrogate_model']).resolve()
    res = settings['projection']['map_resolution']; size = settings['projection']['map_size_pc']
    maps = np.zeros((6,2,res,res)); total = np.zeros((9,6,2)); outside = total.copy()
    mass = np.zeros(9); outside_mass = np.zeros((9,6)); count = 0
    for data in pilot.chunks(paths[part],settings,full,slug,nh,oxygen):
        lum, covered = pilot.predictions(data['actual'],data['slug'],ratios,base,bundle)
        nuclei = data['mass']*prod.MSUN_G*settings['snapshot']['hydrogen_mass_fraction']/prod.MH_G
        light = lum[:,:,1:3]*nuclei[:,None,None]
        region = pilot.region_masks(data['xyz'])
        for r,mask in enumerate(region.T):
            total[r] += light[mask].sum(axis=0); mass[r] += data['mass'][mask].sum()
            for b in range(6):
                bad = mask&~covered[:,b]
                outside[r,b] += light[bad,b].sum(axis=0)
                outside_mass[r,b] += data['mass'][bad].sum()
        for b in range(6):
            for j in range(2):
                maps[b,j] += prod.project_surface(data['xyz'],data['h'],light[:,b,j],size,res)
        count += len(light)
        print(snap,part,'predicted active',count,flush=True)
    sfr, sfr_audit = stellar_maps(paths[part],settings)
    c.design.atomic_npz(target,manifest_sha256=np.asarray(sha),maps=maps,totals=total,
        outside=outside,mass=mass,outside_mass=outside_mass,count=np.asarray(count),sfr=sfr,sfr_audit=sfr_audit)
    return str(target)


def smooth(field, finite=True):
    sigma = 125/(np.sqrt(8*np.log(2))*20)
    value = gaussian_filter(field,sigma,mode='constant',cval=0,truncate=4)
    return value/gaussian_filter(np.ones_like(field),sigma,mode='constant',cval=0,truncate=4) if finite else value


def sfr_slopes(beam,sfr):
    from plot_galic_selection import lattice,sample
    xy = lattice(); radius = np.linalg.norm(xy,axis=1)
    line = np.array([[sample(beam[b,j],xy) for j in range(2)] for b in (0,1)])
    field = (radius>=1)&(radius<8)&np.all(np.isfinite(line)&(line>=0),axis=(0,1))
    rows, slopes = [], []
    edges = np.arange(-3.,-.49,.5)
    for k,window in enumerate((1,10)):
        ss = sample(sfr[k],xy); log = np.full(len(ss),-np.inf)
        np.log10(ss,out=log,where=ss>0)
        for r in range(0,8):
            mask = field if r==0 else field&(radius>=r)&(radius<r+1)
            for lo,hi in zip(edges[:-1],edges[1:]):
                use = mask&(log>=lo)&(log<hi)
                total = line[:,:,use].sum(axis=-1)
                ratio = np.divide(total[:,0],total[:,1],out=np.full(2,np.nan),where=total[:,1]>0)
                rows.append(dict(window_Myr=window,radial_bin=r,log_sfr_lo=lo,log_sfr_hi=hi,
                                 count=int(use.sum()),log_R=np.log10(ratio).tolist()))
        for b,branch in enumerate(('Full','SLUG')):
            for method in ('whole','fixed_effect'):
                numerator=denominator=0.; groups=0
                for r in ([0] if method=='whole' else range(1,8)):
                    selected = [row for row in rows if row['window_Myr']==window and row['radial_bin']==r
                                and row['count']>=10 and np.isfinite(row['log_R'][b])]
                    if len(selected)<(3 if method=='whole' else 2): continue
                    x=np.array([(row['log_sfr_lo']+row['log_sfr_hi'])/2 for row in selected])
                    y=np.array([row['log_R'][b] for row in selected]); dx=x-x.mean()
                    numerator += dx@y; denominator += dx@dx; groups += 1
                slope = numerator/denominator if denominator>0 and groups>=(1 if method=='whole' else 2) else None
                slopes.append(dict(window_Myr=window,branch=branch,method=method,slope=slope))
    return rows,slopes


def merge(snap,ratios,sha):
    root,settings,*_ = environment(snap)
    arrays = {}
    for part in range(8):
        with np.load(root/f'part{part:02d}.npz') as f:
            assert str(f['manifest_sha256']) == sha
            for key in ('maps','totals','outside','mass','outside_mass','count','sfr','sfr_audit'):
                if key not in arrays: arrays[key] = f[key].copy()
                else: arrays[key] += f[key]
    beam = np.array([[smooth(field) for field in model] for model in arrays['maps']])
    sfr = np.array([smooth(field,finite=False) for field in arrays['sfr']])
    masks = pilot.map_masks(settings); integrated = np.zeros((9,6,2))
    for r,mask in enumerate(masks): integrated[r] = beam[:,:,mask].sum(axis=-1)*20**2
    freq = settings['projection']['line_frequencies_Hz']
    factor = (freq['C18O']/freq['13CO'])**3
    ratio = np.divide(integrated[:,:,0],integrated[:,:,1],out=np.full((9,6),np.nan),where=integrated[:,:,1]>0)*factor
    # SFR slopes are unchanged by the spatially constant frequency conversion.
    bins,slopes = sfr_slopes(beam,sfr)
    mass_fraction = np.divide(arrays['outside_mass'],arrays['mass'][:,None],
                              out=np.full((9,6),np.nan),where=arrays['mass'][:,None]>0)
    light_fraction = np.divide(arrays['outside'],arrays['totals'],out=np.full((9,6,2),np.nan),where=arrays['totals']>0)
    entry = next(e for e in json.loads((c.ROOT/'inventory.json').read_text()) if e['snapshot']==snap)
    with np.load(root/'context.npz') as f: oxygen=float(f['oxygen_scale'])
    summary = dict(snapshot=snap,time_Myr=entry['time_Gyr']*1000,manifest_sha256=sha,
        active_count=int(arrays['count']),active_mass_Msun=arrays['mass'][0],oxygen_scale=oxygen,uniform_isotope_ratios=ratios,
        regions=c.REGIONS,models=c.MODELS,beam_signed_luminosity_totals=integrated,R13_18_brightness=ratio,
        source_multiplier=ratio[:,1]/ratio[:,0],exchange_multiplier=ratio[:,2]/ratio[:,3],
        uniform_exchange_multiplier=ratio[:,4]/ratio[:,5],
        coverage_particle_centered_mass_outside=mass_fraction,
        coverage_particle_centered_rare_luminosity_outside=light_fraction,
        coverage_99pct=(mass_fraction<=.01)&np.all(light_fraction<=.01,axis=2),
        sfr_catalogue_audit_columns=['selected','known_positive_targetMass','unknown_targetMass','invalid_age_in_field'],
        sfr_catalogue_audit=arrays['sfr_audit'],sfr_slopes=slopes,sfr_bins=bins,
        scope='Frozen-surrogate temporal diagnostic; 575 direct validation is not an accuracy certificate at other epochs. Coverage is not accuracy.',
        sfr_scope='Surviving PartType4 catalogue, targetMass[:,0] only; windows with missing formation masses are known-mass-only.')
    if snap==575:
        with open(c.ISO/'highres_snapshot575_temporal/output/snapshot575_summary.json') as f:
            old=json.load(f)['total_line_luminosity_erg_s']
        old=np.array([old[k] for k in ('13CO','C18O')]) if isinstance(old,dict) else np.asarray(old)[1:3]
        summary['replay575_production_Full_particle_total_delta_dex']=np.log10(arrays['totals'][0,0]/old)
        summary['replay575_count_matches']=int(arrays['count'])==43133718
    c.design.atomic_npz(root/'beam_maps.npz',manifest_sha256=np.asarray(sha),maps=beam,sfr=sfr)
    c.save(root/'summary.json',summary)
    print('Epoch complete',snap,flush=True)


def main():
    assert int(os.environ.get('PBS_NCPUS','0'))>=104
    sha=c.verify(); snap=int(os.environ['SNAPSHOT']); assert 551<=snap<=575
    root=c.ROOT/f'time/{snap}'
    lock=(root/'epoch.lock').open('a+'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'summary.json').exists():
        assert json.loads((root/'summary.json').read_text())['manifest_sha256']==sha
        return
    settings=prod.load_settings(root/'config.json')
    if not (root/'context.npz').exists(): prod.prepare_context(root/'config.json',settings)
    os.environ['OMP_NUM_THREADS']='13'; os.environ['NUMBA_NUM_THREADS']='13'
    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
        sums=np.sum(list(pool.map(reservoir,[(snap,p) for p in range(8)])),axis=0)
        ratios=[float(sums[0]/sums[1]),float(sums[2]/sums[3])]
        assert np.all(np.isfinite(ratios)) and np.all(np.asarray(ratios)>0)
        c.save(root/'reservoir.json',dict(sums=sums,ratios=ratios,manifest_sha256=sha))
        list(pool.map(predict_part,[(snap,p,ratios,sha) for p in range(8)]))
    merge(snap,ratios,sha)


if __name__=='__main__': main()
