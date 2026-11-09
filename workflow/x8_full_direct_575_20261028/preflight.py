"""PBS-only end-to-end release gate for the four full-population lanes."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np
from core import (ROOT,ISO,BRANCHES,load_plan,verify_code,digest,save_json,read_task,paths,read_state,identity)
import engine
import worker
import tests


def main():
    assert int(os.environ.get('PBS_NCPUS','0'))>0
    target=ROOT/'preflight.json'; assert not target.exists()
    sha=verify_code(); plan=load_plan()
    # Revalidate the cache-generation frozen lineage once, including old models.
    import sys
    sys.path.insert(0,str(ISO/'x8_direct_validation_20261027'))
    import prepare as old_prepare
    assert old_prepare.verify()==plan['parent_manifest_sha256']
    result=dict(complete=False,passed=False,manifest_sha256=sha,tests=tests.run(),cache_sha256={})
    count=0; mass=0.; gas_count=0
    snapshots=json.loads((ROOT/'manifest.json').read_text())['snapshot_identities']
    for entry in plan['parts']:
        with h5py.File(entry['path']) as f:
            assert f.attrs['complete'] and f.attrs['manifest_sha256']==plan['parent_manifest_sha256']
            assert len(f['actual'])==len(f['slug'])==len(f['row'])==len(f['particle_id'])==entry['count']
            assert f['actual'].shape[1]==f['slug'].shape[1]==7
            count+=entry['count']; mass+=float(f.attrs['active_mass_Msun'])
        result['cache_sha256'][entry['path']]=digest(entry['path'])
    for snapshot in snapshots:
        with h5py.File(snapshot['path']) as f: gas_count+=len(f['PartType0/Masses'])
    assert gas_count==plan['total_snapshot_gas_particles']
    assert count==plan['active_count']==43133718
    assert np.isclose(mass,plan['active_mass_Msun'],rtol=1e-10)
    # Verify byte-identical local physics data, not just matching filenames.
    stage=Path(os.environ['FULL575_STAGE'])
    for source,relative in plan['stage_files'].items(): assert digest(source)==digest(stage/relative)
    audit=dict(passed=False,manifest_sha256=sha,cache_sha256=result['cache_sha256'])
    workers=min(16,int(os.environ['PBS_NCPUS']))
    pilot=ISO/'x8_direct_validation_20261027'
    selected=np.linspace(0,127,16,dtype=int)
    replay=[]; expected={}
    with h5py.File(pilot/'pilot_sample.h5') as f:
        for b,branch in enumerate(BRANCHES):
            key='slug' if branch=='slug_on' else 'uniform' if branch.startswith('uniform') else 'actual'
            for i in selected:
                j=len(replay); replay.append((j,branch,tuple(f['unique/'+key][i])))
                old=json.loads((pilot/f'direct_results/{branch}_{i:04d}.json').read_text())['routes'][0]
                expected[j]=dict(converged=old['converged'],values=old['values'],particle=int(i),branch=branch)
    # Uniform-in-population timing probes; these are not selected for brightness.
    rng=np.random.default_rng(2026102803)
    global_rows=rng.choice(count,64,replace=False)
    ends=np.cumsum([p['count'] for p in plan['parts']]); starts=np.r_[0,ends[:-1]]
    probes=[]
    from workflow_config import load_config
    settings=json.loads((ISO/'highres_snapshot575_temporal/config.json').read_text())
    full=load_config(ROOT/'workflow.json')[0]
    slug=load_config(ISO/'highres_snapshot575_temporal/slug_only/source_workflow.json')[0]
    columns,full_plan=old_prepare.prod.isotope_plan(full)
    scols,slug_plan=old_prepare.prod.isotope_plan(slug); columns=sorted(set(columns+scols))
    units=old_prepare.prod.unit_factors(settings)
    with np.load(settings['outputs']['context_npz']) as f: nh_map=f['NH_map_cm2'].copy()
    for g in global_rows:
        part=int(np.searchsorted(ends,g,side='right')); row=int(g-starts[part])
        sample_data=read_task(dict(id=-1,branch='actual_on',part=part,start=row,stop=row+1),plan)
        original=int(sample_data['row'][0]); xh=settings['snapshot']['hydrogen_mass_fraction']
        with h5py.File(snapshots[part]['path']) as f:
            source=f['PartType0']; sl=slice(original,original+1)
            np.testing.assert_array_equal(source['ParticleIDs'][sl],sample_data['particle_id'])
            np.testing.assert_allclose(source['Coordinates'][sl]*units['length_pc'],sample_data['xyz'],rtol=1e-13,atol=1e-10)
            np.testing.assert_allclose(source['Masses'][sl]*units['mass_Msun'],sample_data['mass'],rtol=1e-13)
            n=source['Density'][sl]*units['density_cgs']*xh/old_prepare.prod.MH_G
            dv=np.maximum(source['NormVelocityGradient'][sl]*units['dvdr_km_s_pc'],settings['projection']['minimum_dvdr_km_s_pc'])
            metal=source['Metallicity'][sl,columns]
            size=settings['projection']['map_size_pc']; res=settings['projection']['column_resolution']
            pix=np.floor(sample_data['xyz'][0,:2]/(size/res)).astype(int)+res//2
            col=np.array([nh_map[pix[0],pix[1]]])
            for key,ip in [('actual',full_plan),('slug',slug_plan)]:
                iso=old_prepare.prod.isotope_abundances(metal,columns,ip,xh)
                raw=np.column_stack((n,col,dv,iso['12C'],iso['13C'],iso['16O']*plan['oxygen_scale'],iso['18O']*plan['oxygen_scale']))
                np.testing.assert_allclose(raw,sample_data[key],rtol=1e-13,atol=0)
        for branch in BRANCHES:
            task=dict(id=-1,branch=branch,part=part,start=row,stop=row+1)
            data=read_task(task,plan)
            probes.append((len(replay)+len(probes),branch,tuple(data['raw'][0])))
    comparison=[]; timing=[]; reference_results={}
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=worker.initialize) as pool:
        futures=[pool.submit(worker.solve,t) for t in replay+probes]
        for future in as_completed(futures):
            out=future.result(); j,status=out[:2]
            assert (status==1)==np.all(np.isfinite(out[7]))
            if j<len(replay):
                old=expected[j]; assert (status==1)==old['converged'],('Replay status mismatch',old,out[:7],out[-1])
                if status==1:
                    np.testing.assert_allclose(out[7][:18],old['values'],rtol=1e-10,atol=1e-60)
                reference_results[j]=out
                comparison.append(dict(particle=old['particle'],branch=old['branch'],status=status,attempts=out[2]))
            else: timing.append(dict(branch=(replay+probes)[j][1],status=status,seconds=out[3]))
            if status==1:
                tau=out[7][9:12]; beta=1-tau/2; positive=tau>1e-6
                beta[positive]=-np.expm1(-tau[positive])/tau[positive]
                np.testing.assert_allclose(out[7][18:21],beta,rtol=1e-10,atol=1e-12)
        # Exercise the SAME output/resume engine, with a short real input block.
        task=dict(id=-2,part=0,branch='actual_on',start=10000,stop=10008)
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root=Path(tmp)
            complete,state=engine.process_block(pool,task,plan,audit,root=root,stop_after=2)
            assert not complete and np.count_nonzero(state['status'])==2
            first=state['values'].copy(); statuses=state['status'].copy()
            complete,state=engine.process_block(pool,task,plan,audit,root=root)
            assert complete and np.all(state['status']!=0)
            np.testing.assert_array_equal(first[statuses!=0],state['values'][statuses!=0])
            final,_=paths(task,root)
            data=read_task(task,plan); metadata=identity(task,data,plan,audit)
            saved=read_state(final,metadata,8,require_complete=True,data=data)
            np.testing.assert_array_equal(saved['status'],state['status'])
            class NoSubmit:
                def submit(self,*args): raise AssertionError('A completed block was recomputed')
            assert engine.process_block(NoSubmit(),task,plan,audit,root=root)[0]
    # New processes, original data paths: staging must not alter the solution.
    subset=[t for t in replay if reference_results[t[0]][1]==1][:8]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),
            initializer=worker.initialize,initargs=(True,False)) as pool:
        for out in pool.map(worker.solve,subset):
            assert out[1]==1
            np.testing.assert_allclose(out[7],reference_results[out[0]][7],rtol=1e-10,atol=1e-60)
            np.testing.assert_allclose(out[8],reference_results[out[0]][8],rtol=1e-10,atol=1e-60)
    mean_seconds=float(np.mean([r['seconds'] for r in timing]))
    result.update(complete=True,passed=True,active_count=count,active_mass_Msun=mass,
        replay_cases=len(replay),replay=comparison,uniform_timing_cases=len(probes),
        uniform_timing=timing,mean_primary_state_worker_seconds=mean_seconds,
        rough_full_five_branch_worker_hours=mean_seconds*plan['physical_states']/3600,
        actual_beta_checked=True,random_snapshot_input_reconstruction_passed=True,
        real_checkpoint_resume_passed=True,byte_identical_staging_passed=True,
        note='Uniform timing probes are a small planning estimate; not an accuracy certificate or a guaranteed completion time.')
    # Publish the release gate only after every check succeeds.
    save_json(target,result)
    audit['passed']=True; audit['preflight_sha256']=digest(target)
    save_json(ROOT/'input_audit.json',audit)
    print(json.dumps({k:v for k,v in result.items() if k not in ('replay','uniform_timing','cache_sha256','tests')},indent=2),flush=True)


if __name__=='__main__': main()
