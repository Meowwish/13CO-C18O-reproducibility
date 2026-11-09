"""Construct and check the whole-population registry, then freeze its inputs."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
from workflow_config import load_config,config_path
from core import ROOT,ISO,BRANCHES,digest,save_json


def main():
    assert not (ROOT/'submissions.json').exists()
    if (ROOT/'manifest.json').exists():
        assert '--refresh-before-submission' in sys.argv
        assert not list((ROOT/'output').rglob('*.npz')) and not (ROOT/'preflight.json').exists()
        (ROOT/'manifest.pre_submission_review.json').write_bytes((ROOT/'manifest.json').read_bytes())
        (ROOT/'manifest.json').unlink()
    parent=ISO/'x8_direct_validation_20261027'
    parent_sha=digest(parent/'manifest.json')
    old_manifest=json.loads((parent/'manifest.json').read_text())
    cfg=copy.deepcopy(load_config(ISO/'chemTable_temporal_550_575/config/snapshot575.json')[0])
    cfg['run_name']='full_x8_snapshot575_primary_auto_20261028'
    cfg['description']='Entire active x8/575 population; primary auto timescale; nominal labels with explicit failures.'
    cfg['paths'].update(snapshot=old_manifest['snapshots'][0]['path'],direct_input=str(parent/'cache/part00.h5'),
        retry_input=str(ROOT/'UNUSED_retry.h5'),primary_chunk_dir=str(ROOT/'output'),retry_chunk_dir=str(ROOT/'UNUSED_retry'),
        merged_result=str(ROOT/'UNUSED_merged.h5'),map_output_dir=str(ROOT/'UNUSED_maps'))
    assert cfg['chemistry']['setChemEq']['tEqGuess_s'] is None
    assert cfg['chemistry']['setChemEq']['small_abundance']==1e-15
    assert cfg['chemistry']['setChemEq']['relative_tolerance']==.001
    save_json(ROOT/'workflow.json',cfg)
    assert load_config(ROOT/'workflow.json')[0]==cfg
    context=json.loads((ISO/'isotope_fractionation_controls/x8_snapshot575/context.json').read_text())
    parts=[]
    for p in range(8):
        path=parent/f'cache/part{p:02d}.h5'; s=path.stat()
        with h5py.File(path) as f:
            assert f.attrs['complete'] and f.attrs['manifest_sha256']==parent_sha
            count=len(f['actual'])
            assert count==int(f.attrs['active_count']) and count==len(f['slug'])
            parts.append(dict(path=str(path),size=s.st_size,mtime_ns=s.st_mtime_ns,count=count,
                mass_Msun=float(f.attrs['active_mass_Msun'])))
    total=sum(p['count'] for p in parts); mass=sum(p['mass_Msun'] for p in parts)
    assert total==context['active_count']==43133718
    assert np.isclose(mass,context['active_mass_Msun'],rtol=1e-10)
    tasks=[]
    for branch in BRANCHES:
        for p,entry in enumerate(parts):
            for start in range(0,entry['count'],4096):
                tasks.append(dict(id=len(tasks),branch=branch,part=p,start=start,stop=min(start+4096,entry['count'])))
    stage_files={str(config_path(cfg,'cloud_file')):'MilkyWayGMC.desp'}
    for source_root,prefix in [(ISO/'chemTable_v4_full_direct/LAMDA','LAMDA'),(config_path(cfg,'visser_shielding_dir'),'visser')]:
        for p in source_root.rglob('*'):
            if p.is_file(): stage_files[str(p)]=str(Path(prefix)/p.relative_to(source_root))
    plan=dict(snapshot=575,resolution='x8',active_count=total,total_snapshot_gas_particles=126192499,
        active_mass_Msun=mass,physical_states=total*5,parts=parts,tasks=tasks,task_count=len(tasks),
        parent_manifest_sha256=parent_sha,uniform_ratios=context['ratios'],oxygen_scale=context['oxygen_scale'],
        stage_files=stage_files,lanes=4,block_size=4096,checkpoint_every=128,primary_tEqGuess=None,
        primary_smallabd=1e-15,primary_tol=.001,first_wave_cpu_limit=416,
        first_wave_max_SU=39936,expected_raw_numerical_output_GB=total*5*47*8/1e9,
        failed_value_policy='NaN; retained status and error; no projection or training in this campaign')
    save_json(ROOT/'registry.json',plan)
    from tests import run
    verification=run()
    save_json(ROOT/'local_verification.json',verification)
    for p in ROOT.glob('*.py'): ast.parse(p.read_text(),filename=str(p))
    subprocess.run(['bash','-n',str(ROOT/'run.pbs')],check=True)
    physics=ISO/'chemTable_v4_full_direct'
    files=[*ROOT.glob('*.py'),ROOT/'PLAN.md',ROOT/'run.pbs',ROOT/'workflow.json',ROOT/'registry.json',ROOT/'local_verification.json',
        physics/'scripts/solve_direct_chunk.py',physics/'scripts/workflow_config.py',parent/'manifest.json',
        parent/'pilot_sample.h5',ISO/'chemTable_temporal_550_575/config/snapshot575.json',
        ISO/'chemTable_v6_brightness_validation/config/snapshot600_face_on_tight.json',
        ISO/'chemTable_v4_full_direct/config/snapshot600_full_direct.json',
        ISO/'isotope_fractionation_controls/x8_snapshot575/context.json',*map(Path,stage_files)]
    runtime=physics/'despotic_runtime/despotic'
    files+=list(runtime.rglob('*.py'))+list(runtime.glob('*.so'))
    for b in BRANCHES:
        for i in np.linspace(0,127,16,dtype=int): files.append(parent/f'direct_results/{b}_{i:04d}.json')
    manifest=dict(hashes={str(p):digest(p) for p in sorted(set(files))},
        cache_identities=[{k:v for k,v in p.items() if k in ('path','size','mtime_ns')} for p in parts],
        snapshot_identities=old_manifest['snapshots'],manuscript_sha256=digest(ISO/'Paper/mnras_template.tex'),
        python_version=sys.version,auto_time_guess=True,full_target_states=total*5)
    save_json(ROOT/'manifest.json',manifest)
    print(json.dumps(dict(active_particles=total,physical_states=total*5,blocks=len(tasks),
        lanes=4,raw_output_GB=plan['expected_raw_numerical_output_GB'],frozen_files=len(manifest['hashes']),
        tests_passed=True,manifest_sha256=digest(ROOT/'manifest.json')),indent=2),flush=True)


if __name__=='__main__': main()
