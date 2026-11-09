"""Prepare, submit, retry and merge temporal direct reference calculations."""
import json
import os
from pathlib import Path
import subprocess
import sys
import h5py

ROOT = Path(__file__).resolve().parent
V4 = ROOT.parent / 'chemTable_v4_full_direct'
V6 = ROOT.parent / 'chemTable_v6_brightness_validation'
config = Path(os.environ['CHEMTABLE_V4_CONFIG'])
stage = os.environ.get('TEMPORAL_STAGE', 'prepare')
number = json.loads(config.read_text())['snapshot']['index']
branch = ROOT / ('snapshot' + str(number))

def run(script, *args):
    subprocess.run([sys.executable, '-u', str(V4 / 'scripts' / script), *args], check=True)

def next_stage(name, dependencies):
    subprocess.run(['qsub', '-W', 'depend=afterok:' + ':'.join(dependencies),
                    '-v', 'CHEMTABLE_V4_CONFIG=' + str(config) + ',TEMPORAL_STAGE=' + name,
                    str(ROOT / 'pbs/pipeline.pbs')], check=True)

if stage == 'merge':
    run('merge_direct_results.py')
    print('Merged temporal reference:', number, flush=True)
else:
    mode = 'primary' if stage == 'prepare' else 'retry'
    run('prepare_snapshot_inputs.py' if mode == 'primary' else 'prepare_failed_retry_inputs.py')
    if mode == 'retry':
        with h5py.File(branch / 'input' / ('snapshot_' + str(number) + '_failed_retry_inputs.h5')) as source:
            count = int(source.attrs['active_particle_count'])
        if count == 0:
            next_stage('merge', [os.environ['PBS_JOBID']])
            sys.exit(0)
    manifest = branch / 'output' / (mode + '_submissions.json')
    run('submit_direct_arrays.py', '--prepare-job', os.environ['PBS_JOBID'],
        '--mode', mode, '--config', str(config),
        '--pbs-script', str(V6 / 'pbs/solve_shielding_array.pbs'),
        '--manifest', str(manifest))
    records = json.loads(manifest.read_text())
    if not records['all_submissions_succeeded']:
        raise RuntimeError('Incomplete array submission')
    jobs = [item['stdout'] for item in records['submissions']]
    next_stage('retry' if mode == 'primary' else 'merge', jobs)
