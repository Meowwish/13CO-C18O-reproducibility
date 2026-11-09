"""Paired isotope-exchange pilot on original-resolution cached inputs only."""
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
ISO = ROOT.parent
CONFIG_PATH = Path(os.environ.get('CONTROL_CONFIG', ROOT / 'config.json')).resolve()
CFG = json.loads(CONFIG_PATH.read_text())
NAMES = ('nH_cm3', 'NH_cm2', 'dVdr_km_s_pc', 'x12C', 'x13C', 'x16O', 'x18O')


def transform(rows, c_ratio, o_ratio):
    result = rows.copy()
    for main, rare, ratio in ((3, 4, c_ratio), (5, 6, o_ratio)):
        total = rows[:, main] + rows[:, rare]
        result[:, rare] = total / (1 + ratio)
        result[:, main] = total - result[:, rare]
        np.testing.assert_allclose(result[:, main] + result[:, rare], total, rtol=1e-14)
    return result


def solve(task):
    import solve_direct_chunk as direct
    index, branch, row = task
    direct.CHEMISTRY['include_13C_fractionation'] = branch.endswith('_on')
    return branch, direct.solve_one((index, row))


def prepare(number):
    path = Path(CFG.get('input_paths', {}).get(str(number),
        ISO / f'chemTable_temporal_550_575/snapshot{number}/input/snapshot_{number}_direct_inputs.h5'))
    with h5py.File(path) as f:
        if not bool(f.attrs.get('complete', False)):
            raise RuntimeError('Incomplete cached input')
        rows = np.column_stack([f[k][:] for k in NAMES])
        mass = f['mass_Msun'][:]
        radius = np.hypot(*f['coordinates_pc'][:, :2].T) / 1000
        ids = f['particle_index'][:]
    valid = np.all(np.isfinite(rows) & (rows > 0), axis=1) & np.isfinite(mass) & (mass > 0) & np.isfinite(radius)
    sums = np.sum(rows[valid, 3:] * mass[valid, None], axis=0)
    ratios = sums[0] / sums[1], sums[2] / sums[3]
    codes = np.column_stack((np.digitize(radius, CFG['radius_internal_edges_kpc']),
                            np.digitize(np.log10(np.maximum(rows[:, 0], 1e-300)), CFG['log_nH_internal_edges']),
                            np.digitize(np.log10(np.maximum(rows[:, 1], 1e-300)), CFG['log_NH_internal_edges'])))
    eligible = np.flatnonzero(valid)
    _, group = np.unique(codes[valid], axis=0, return_inverse=True)
    rng = np.random.default_rng(CFG['seed'] + number)
    selected, probability = [], []
    for code in np.unique(group):
        pool = eligible[group == code]
        count = min(len(pool), CFG['samples_per_stratum'])
        selected.extend(rng.choice(pool, count, replace=False))
        probability.extend([count / len(pool)] * count)
    selected = np.asarray(selected)
    return rows[selected], dict(source_row=selected, particle_index=ids[selected],
        mass_Msun=mass[selected], radius_kpc=radius[selected], stratum=codes[selected],
        inclusion_probability=np.asarray(probability)), dict(
            input=str(path), input_size=path.stat().st_size, input_mtime_ns=path.stat().st_mtime_ns,
            eligible_count=int(valid.sum()), excluded_count=int((~valid).sum()),
            excluded_mass_fraction=float(mass[~valid & np.isfinite(mass) & (mass > 0)].sum() / mass[np.isfinite(mass) & (mass > 0)].sum()),
            c12_c13=float(ratios[0]), o16_o18=float(ratios[1]))


def main():
    import solve_direct_chunk as direct
    out = ROOT / CFG.get('output_subdir', 'output')
    out.mkdir(exist_ok=True)
    number = int(os.environ['CONTROL_SNAPSHOT'])
    assert number in CFG['snapshots']
    assert direct.SMALL_ABUNDANCE == 1e-15
    assert direct.SOLVER['tEqGuess_s'] is None
    assert direct.CLOUD['cosmic_ray_ionization_rate_s-1'] == 1e-16
    assert direct.CONFIG['snapshot']['index'] == number
    rows, metadata, provenance = prepare(number)
    if 'validation_fraction' in CFG:
        # A particle and all its interventions share one preassigned partition.
        split_rng = np.random.default_rng(CFG['partition_seed'] + number)
        _, groups = np.unique(metadata['stratum'], axis=0, return_inverse=True)
        validation = np.zeros(len(rows), dtype=np.int8)
        for group in np.unique(groups):
            pool = np.flatnonzero(groups == group)
            if len(pool) >= 8:
                count = max(1, int(np.floor(len(pool) * CFG['validation_fraction'])))
                validation[split_rng.choice(pool, count, replace=False)] = 1
        metadata['validation_partition'] = validation
    uniform = transform(rows, provenance['c12_c13'], provenance['o16_o18'])
    retry = direct.RUN_MODE == 'retry'
    suffix = '_retry' if retry else ''
    previous = out / f'snapshot{number}_paired.h5'
    cached = {}
    if retry:
        with h5py.File(previous) as prior:
            if not bool(prior.attrs.get('complete', False)):
                raise RuntimeError('Primary pilot is incomplete')
            np.testing.assert_array_equal(prior['actual_inputs'][:], rows)
            np.testing.assert_array_equal(prior['uniform_inputs'][:], uniform)
            np.testing.assert_array_equal(prior['source_row'][:], metadata['source_row'])
            if 'validation_partition' in metadata:
                np.testing.assert_array_equal(prior['validation_partition'][:], metadata['validation_partition'])
            for branch in CFG['branches']:
                cached[branch] = {key: prior[branch + '/' + key][:]
                                  for key in ('status', 'attempts', 'values', 'error')}
        provenance['primary_result'] = str(previous)
    provenance['stage'] = direct.RUN_MODE
    provenance['resolved_workflow'] = direct.CONFIG
    provenance['pilot_config'] = CFG
    import importlib
    network = Path(importlib.import_module('despotic.chemistry.GOW').__file__)
    provenance['network_file'] = str(network)
    provenance['sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), CONFIG_PATH, network, Path(direct.__file__))}
    path = out / f'snapshot{number}_paired{suffix}.h5'
    # Refuse overwriting partial or completed results: preserve the audit trail.
    with h5py.File(path, 'x') as f:
        f.attrs['complete'] = False
        f.attrs['provenance_json'] = json.dumps(provenance)
        f.attrs['output_names_json'] = json.dumps(direct.OUTPUT_NAMES)
        f.create_dataset('actual_inputs', data=rows)
        f.create_dataset('uniform_inputs', data=uniform)
        for key, value in metadata.items():
            f.create_dataset(key, data=value)
        for branch in CFG['branches']:
            g = f.create_group(branch)
            g.create_dataset('status', data=np.zeros(len(rows), dtype='i1'))
            g.create_dataset('attempts', data=np.zeros(len(rows), dtype='i2'))
            g.create_dataset('values', data=np.full((len(rows), len(direct.OUTPUT_NAMES)), np.nan))
            g.create_dataset('error', shape=(len(rows),), dtype=h5py.string_dtype())
            if retry:
                for key, value in cached[branch].items():
                    g[key][:] = value
        tasks = [(i, branch, tuple(row)) for branch in CFG['branches']
                 for i, row in enumerate(uniform if branch.startswith('uniform') else rows)
                 if not retry or cached[branch]['status'][i] != 1]
        workers = min(CFG['workers'], int(os.environ.get('PBS_NCPUS', '1')))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(solve, task) for task in tasks]
            for count, future in enumerate(as_completed(futures), 1):
                branch, (i, state, attempts, values, error) = future.result()
                f[branch + '/status'][i] = state
                f[branch + '/attempts'][i] = attempts
                f[branch + '/values'][i] = values
                f[branch + '/error'][i] = error
                if count % 16 == 0 or count == len(tasks):
                    f.flush()
                    print(f'{number}: {count}/{len(tasks)}', flush=True)
        f.attrs['complete'] = True
        f.flush()
    with h5py.File(path) as f:
        good = np.logical_and.reduce([f[b + '/status'][:] == 1 for b in CFG['branches']])
        values = {b: f[b + '/values'][:] for b in CFG['branches']}
    summary = dict(snapshot=number, selected=len(rows), common_converged=int(good.sum()),
                   failure_counts={b:int(np.sum(~np.all(np.isfinite(v), axis=1))) for b,v in values.items()},
                   warning='Stratified pilot medians are NOT galaxy-integrated effects; only common converged positive pairs are compared.')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    effects = {}
    for ax, (numerator, denominator, title) in zip(axes, [(4, 5, 'Molecular 13CO/C18O'), (7, 8, 'Line L13/L18')]):
        for mixture in ('actual', 'uniform'):
            on, off = values[mixture + '_on'], values[mixture + '_off']
            use = good & (on[:, numerator] > 0) & (on[:, denominator] > 0) & (off[:, numerator] > 0) & (off[:, denominator] > 0)
            delta = np.log10(on[use, numerator] / on[use, denominator]) - np.log10(off[use, numerator] / off[use, denominator])
            ax.scatter(on[use, 0], delta, s=8, alpha=.5, label=mixture)
            effects[title + '_' + mixture] = dict(count=int(use.sum()),
                median_dex=float(np.median(delta)) if len(delta) else None)
        ax.axhline(0, color='.4', ls=':')
        ax.set(xscale='log', xlabel='Exchange-on gas temperature [K]',
               ylabel='log10(exchange on / off) [dex]', title=title)
        ax.legend()
    fig.suptitle(f'Original snapshot {number}: paired pilot, not a galaxy average')
    for ext in ('png', 'pdf'):
            fig.savefig(out / f'snapshot{number}_exchange_pilot{suffix}.{ext}', dpi=180)
    plt.close(fig)
    summary['exchange_effects'] = effects
    (out / f'snapshot{number}_summary{suffix}.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
