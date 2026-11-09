"""Four paired frozen-surrogate interventions for x8/575; no direct chemistry."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import process_snapshot563_lookup_v11 as production
from workflow_config import load_config
from adaptive_local_surrogate import (AdaptiveLocalActualAbundanceSurrogate, OUTPUT_NAMES,
    raw_inputs_to_features,transform_outputs,inverse_transform_outputs)
from plot_galic_selection import lattice,sample

ROOT=Path(__file__).resolve().parent
ISO=ROOT.parent
OUT=ROOT/'x8_snapshot575'
BRANCHES=('actual_on','actual_off','uniform_on','uniform_off')
SETTINGS_PATH=ISO/'highres_snapshot575_temporal/config.json'


def environment():
    settings=production.load_settings(SETTINGS_PATH)
    paths=production.snapshot_files(settings)
    production.validate_snapshot_header(paths,575)
    source=load_config(settings['chemistry']['source_workflow_config'])[0]
    assert source['cloud']['cosmic_ray_ionization_rate_s-1']==1e-16
    assert source['chemistry']['setChemEq']['small_abundance']==1e-15
    assert source['chemistry']['setChemEq']['tEqGuess_s'] is None
    assert source['isotope_fields']['oxygen_renormalization']['apply_same_factor_to_18O']
    context=Path(settings['outputs']['context_npz'])
    frozen=json.loads((ROOT/'config_final600.json').read_text())
    test_path=ROOT/'final_test_600/frozen_model_final_evaluation.json'
    test=json.loads(test_path.read_text())
    assert test['model_sha256']==frozen['frozen_model_sha256']
    assert test['metrics']['selection_screen_passed']
    model_path=Path(frozen['frozen_model'])
    assert production.file_sha256(model_path)==frozen['frozen_model_sha256']
    signature=dict(settings_sha256=production.file_sha256(SETTINGS_PATH),
        source_config=source,model_sha256=frozen['frozen_model_sha256'],
        final_test_sha256=production.file_sha256(test_path),
        context_sha256=production.file_sha256(context),script_sha256=production.file_sha256(Path(__file__)),
        snapshots=[dict(path=str(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in paths])
    with np.load(context) as f:
        nh=f['NH_map_cm2'].copy();oxygen=float(f['oxygen_scale'])
    return settings,paths,source,nh,oxygen,signature,model_path


def particles(path,settings,source,nh,oxygen):
    units=production.unit_factors(settings);snap=settings['snapshot'];proj=settings['projection']
    columns,plan=production.isotope_plan(source)
    xh=snap['hydrogen_mass_fraction'];res=proj['column_resolution'];pixel=proj['map_size_pc']/res
    with h5py.File(path) as f:
        g=f['PartType0']
        for start in range(0,len(g['Masses']),250000):
            sl=slice(start,start+250000)
            xyz=g['Coordinates'][sl]*units['length_pc'];h=g['SmoothingLength'][sl]*units['length_pc']
            mass=g['Masses'][sl]*units['mass_Msun']
            n=g['Density'][sl]*units['density_cgs']*xh/production.MH_G
            dv=np.maximum(g['NormVelocityGradient'][sl]*units['dvdr_km_s_pc'],proj['minimum_dvdr_km_s_pc'])
            hii=g['GrackleHII'][sl];hii[g['InternalEnergy'][sl]>snap['hot_gas_internal_energy_threshold']]=xh
            iso=production.isotope_abundances(g['Metallicity'][sl,columns],columns,plan,xh)
            x12,x13,x16,x18=iso['12C'],iso['13C'],iso['16O']*oxygen,iso['18O']*oxygen
            pix=np.floor(xyz[:,:2]/pixel).astype(int)+res//2
            footprint=np.all((pix>=0)&(pix<res),axis=1)
            col=np.zeros(len(mass));col[footprint]=nh[pix[footprint,0],pix[footprint,1]]
            raw=np.column_stack((n,col,dv,x12,x13,x16,x18))
            use=footprint & (hii<=snap['maximum_neutral_hii_fraction'])
            use &= np.isfinite(mass)&(mass>0)&np.isfinite(h)&(h>0)
            use &= np.all(np.isfinite(raw),axis=1)&np.all(raw[:,[0,1,2,3,5]]>0,axis=1)&np.all(raw[:,[4,6]]>=0,axis=1)
            if use.any():
                yield raw[use],mass[use],xyz[use],h[use]


def uniform(raw,ratios):
    result=raw.copy()
    for a,b,r in ((3,4,ratios[0]),(5,6,ratios[1])):
        total=raw[:,a]+raw[:,b];result[:,b]=total/(1+r);result[:,a]=total-result[:,b]
    return result


def main(stage,part):
    OUT.mkdir(exist_ok=True)
    settings,paths,source,nh,oxygen,signature,model_path=environment()
    context_path=OUT/'context.json'
    if stage=='context':
        if context_path.exists():
            assert json.loads(context_path.read_text())['signature']==signature
            return
        sums=np.zeros(4);mass_sum=0.;count=0
        for path in paths:
            for raw,mass,_,_ in particles(path,settings,source,nh,oxygen):
                sums+=np.sum(mass[:,None]*raw[:,3:],axis=0);mass_sum+=mass.sum();count+=len(mass)
            print('Uniform reservoirs:',path,flush=True)
        ratios=[float(sums[0]/sums[1]),float(sums[2]/sums[3])]
        baseline=json.loads((ISO/'highres_snapshot575_temporal/output/snapshot575_summary.json').read_text())
        assert count==baseline['active_particle_count'], 'Active selection differs from production'
        context_path.write_text(json.dumps(dict(signature=signature,ratios=ratios,
            active_mass_Msun=mass_sum,active_count=count,oxygen_scale=oxygen,
            definition='Ratios of total isotope number reservoirs over production active gas; per-particle total C/O number budgets preserved.'),indent=2)+'\n')
        return
    context=json.loads(context_path.read_text());assert context['signature']==signature
    resolution=settings['projection']['map_resolution'];size=settings['projection']['map_size_pc']
    if stage=='part':
        assert 0<=part<len(paths)
        target=OUT/f'part{part:02d}.npz'
        if target.exists():
            with np.load(target) as f: assert json.loads(str(f['signature']))==signature
            return
        bundle=joblib.load(model_path)
        assert production.file_sha256(Path(bundle['base_model']))==bundle['base_sha256']
        base=AdaptiveLocalActualAbundanceSurrogate(bundle['base_model'],require_validated=False)
        maps={b:np.zeros((6,resolution,resolution)) for b in BRANCHES}
        totals={b:np.zeros(8) for b in BRANCHES} # mass, outside mass, three L, three outside L
        count=0
        for raw,mass,xyz,h in particles(paths[part],settings,source,nh,oxygen):
            count+=len(mass);nuclei=mass*production.MSUN_G*settings['snapshot']['hydrogen_mass_fraction']/production.MH_G
            for mixture in ('actual','uniform'):
                r=raw if mixture=='actual' else uniform(raw,context['ratios'])
                initial,diag=base.query(r,chunk_size=100000)
                bt=transform_outputs(r,np.column_stack([initial[n] for n in OUTPUT_NAMES]))
                features=raw_inputs_to_features(r);aug=np.column_stack((features,bt))
                for physics in ('on','off'):
                    b=mixture+'_'+physics
                    correction=np.column_stack([m.predict(aug) for m in bundle['models'][physics]])
                    pred=inverse_transform_outputs(r,bt+correction,bundle['target_clips'][physics])
                    assert np.all(np.isfinite(pred)&(pred>=0))
                    bounds=bundle['feature_bounds'][physics]
                    covered=np.all((features>=bounds['min'])&(features<=bounds['max']),axis=1)
                    covered &= diag['within_axis_bounds']&diag['within_joint_coverage']
                    lum=nuclei[:,None]*pred[:,3:]
                    totals[b]+=np.r_[mass.sum(),mass[~covered].sum(),lum.sum(axis=0),lum[~covered].sum(axis=0)]
                    weights=np.column_stack((mass[:,None]*pred[:,:3],lum))
                    for j in range(6):
                        maps[b][j]+=production.project_surface(xyz,h,weights[:,j],size,resolution)
            print(f'part {part}: processed {count} active particles',flush=True)
        production.atomic_write_npz(target,signature=np.asarray(json.dumps(signature)),count=np.asarray(count),
            **{b+'_maps':maps[b] for b in BRANCHES},**{b+'_totals':totals[b] for b in BRANCHES})
        return
    maps={b:np.zeros((6,resolution,resolution)) for b in BRANCHES};totals={b:np.zeros(8) for b in BRANCHES};count=0
    for i in range(len(paths)):
        with np.load(OUT/f'part{i:02d}.npz') as f:
            assert json.loads(str(f['signature']))==signature
            count+=int(f['count'])
            for b in BRANCHES:
                maps[b]+=f[b+'_maps'];totals[b]+=f[b+'_totals']
    assert count==context['active_count']
    coverage={}
    for b,t in totals.items():
        assert np.isclose(t[0],context['active_mass_Msun'],rtol=1e-10)
        coverage[b]=dict(mass_fraction=1-t[1]/t[0],line_fractions=(1-t[5:]/t[2:5]).tolist())
    passed=all(c['mass_fraction']>=.99 and min(c['line_fractions'][1:])>=.99 for c in coverage.values())
    summary=dict(context=context,coverage=coverage,coverage_passed=passed,active_count=count,
        scope='Four Full-source interventions at x8/575; independent original600 particle-summed screening passed, no pixel-level closure claim.')
    # Retain diagnostics even if the transfer gate fails; never silently remove uncovered gas.
    production.atomic_write_npz(OUT/'maps_native.npz',**maps)
    beam={};pixel=size/resolution;sigma=125/(np.sqrt(8*np.log(2))*pixel)
    freq=settings['projection']['line_frequencies_Hz']
    for b in BRANCHES:
        sm=np.array([gaussian_filter(v,sigma,mode='constant',cval=0) for v in maps[b]])
        for j,name in enumerate(('CO','13CO','C18O')):
            beam[b+'_W'+name]=production.luminosity_to_w(sm[j+3],freq[name])
            beam[b+'_X'+name]=sm[j]
    production.atomic_write_npz(OUT/'maps_125pc.npz',**beam)
    xy=lattice();radius=np.hypot(*xy.T)
    fields={b:(sample(beam[b+'_W13CO'],xy),sample(beam[b+'_WC18O'],xy)) for b in BRANCHES}
    masks={b:(v[0]>3*.000163)&(v[1]>3*.000144) for b,v in fields.items()}
    common=np.logical_and.reduce(list(masks.values()))
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained');records=[]
    for ax,mode in zip(axes,('uncut','own_D1000','common_D1000')):
        for b,(a,d) in fields.items():
            selection=np.ones(len(radius),dtype=bool) if mode=='uncut' else masks[b] if mode=='own_D1000' else common
            y=[]
            for lo in range(15):
                use=selection&(radius>=lo)&(radius<lo+1)
                value=float(a[use].sum()/d[use].sum()) if use.any() and d[use].sum()>0 else np.nan
                y.append(value);records.append(dict(branch=b,mode=mode,r_lo=lo,ratio=value,count=int(use.sum())))
            ax.plot(np.arange(15)+.5,y,'o-',ms=3,label=b)
        ax.set(xlabel='Radius [kpc]',ylabel='Sum W13 / sum W18',title=mode,xlim=(0,15))
        ax.axvspan(0,1,color='.93');ax.axvspan(8,15,color='.93');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('x8 snapshot575 | '+('coverage passed' if passed else 'DIAGNOSTIC ONLY: coverage failed'))
    for ext in ('png','pdf'): fig.savefig(OUT/f'radial_controls.{ext}',dpi=180)
    plt.close(fig)
    summary['radial']=records
    summary['integrated_ratios']={b:float(beam[b+'_W13CO'].sum()/beam[b+'_WC18O'].sum()) for b in BRANCHES}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(11,10),layout='constrained')
    for ax,b in zip(axes.flat,BRANCHES):
        a,d=fields[b];use=masks[b]
        ax.set_facecolor('.9')
        im=ax.scatter(*xy[use].T,c=a[use]/d[use],s=2,marker='h',linewidths=0,
            cmap='viridis',vmin=0,vmax=40,rasterized=True)
        ax.set(xlabel='x [kpc]',ylabel='y [kpc]',title=b,xlim=(-15,15),ylim=(-15,15),aspect='equal')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='W13 / W18',extend='max')
    fig.suptitle('125 pc | dual S/N > 3 | D=1000 | '+('coverage passed' if passed else 'coverage failed'))
    for ext in ('png','pdf'):fig.savefig(OUT/f'ratio_maps.{ext}',dpi=180)
    plt.close(fig)
    print(json.dumps(dict(coverage_passed=passed,coverage=coverage,ratios=summary['integrated_ratios'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('context','part','merge'));p.add_argument('--part',type=int,default=0)
    a=p.parse_args();main(a.stage,a.part)
