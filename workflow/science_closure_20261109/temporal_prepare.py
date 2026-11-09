"""Independent per-time radius/density SRS using bounded priority reservoirs."""
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
import multiprocessing
import os

import h5py
import numpy as np

import closure_common as c
import process_snapshot563_lookup_v11 as prod
from workflow_config import load_config

spec = importlib.util.spec_from_file_location('closure_input_reader', c.ISO/'x8_direct_validation_20261027/prepare.py')
reader = importlib.util.module_from_spec(spec); spec.loader.exec_module(reader)


def draw_part(task):
    snap, part = task
    folder = c.ROOT/f'temporal{snap}'
    source = c.ISO/f'x8_science_extension_20261102/time/{snap}'
    settings = prod.load_settings(source/'config.json')
    paths = prod.snapshot_files(settings)
    prod.validate_snapshot_header(paths, snap)
    full = load_config(settings['chemistry']['source_workflow_config'])[0]
    slug = load_config(c.ISO/'highres_snapshot575_temporal/slug_only/source_workflow.json')[0]
    with np.load(source/'context.npz') as f:
        nh, oxygen = f['NH_map_cm2'].copy(), float(f['oxygen_scale'])
    rng = np.random.default_rng(c.SEED+snap*100+part)
    populations = np.zeros(48, 'i8'); saved = {}
    fields = ('actual', 'slug', 'mass', 'xyz', 'h', 'row', 'particle_id')
    for data in reader.chunks(paths[part], settings, full, slug, nh, oxygen):
        radius = np.digitize(np.hypot(*data['xyz'][:, :2].T)/1000, np.arange(1, 16))
        strata = radius*3+np.digitize(np.log10(data['actual'][:, 0]), [1, 3])
        keys = rng.random(len(strata))
        populations += np.bincount(strata, minlength=48)
        for h in np.unique(strata):
            ix = np.flatnonzero(strata == h)
            block = {k:data[k][ix] for k in fields}
            block.update(priority=keys[ix], stratum=np.full(len(ix), h), part=np.full(len(ix), part))
            if h in saved:
                block = {k:np.concatenate((saved[h][k], block[k])) for k in block}
            keep = np.argsort(block['priority'])[:c.TARGETS[h//3]]
            saved[h] = {k:v[keep] for k, v in block.items()}
    target = folder/f'reservoir{part:02d}.npz'
    assert not target.exists()
    packed = {k:np.concatenate([saved[h][k] for h in sorted(saved)]) for k in next(iter(saved.values()))}
    c.parent.base.atomic_npz(target, population=populations, **packed)
    return str(target)


def main():
    assert int(os.environ.get('PBS_NCPUS', '0')) > 0
    manifest = c.verify(); snap = int(os.environ['SNAPSHOT']); assert snap in c.SNAPSHOTS
    folder = c.ROOT/f'temporal{snap}'; folder.mkdir(exist_ok=True)
    assert not (folder/'sample.h5').exists()
    with ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context('spawn')) as pool:
        paths = list(pool.map(draw_part, [(snap, part) for part in range(8)]))
    blocks = []; population = np.zeros(48, 'i8')
    for path in paths:
        with np.load(path) as f:
            population += f['population']
            blocks.append({k:f[k].copy() for k in f.files if k != 'population'})
    all_data = {k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    allocation = np.zeros(48, 'i8'); selected = []
    for radial in range(16):
        s = slice(3*radial, 3*radial+3); N = population[s]
        n = c.parent.base.allocation(N, min(int(c.TARGETS[radial]), int(N.sum())))
        allocation[s] = n
        for h in range(3*radial, 3*radial+3):
            ids = np.flatnonzero(all_data['stratum'] == h)
            selected.extend(ids[np.argsort(all_data['priority'][ids])[:allocation[h]]])
    selected = np.asarray(selected)
    data = {k:v[selected] for k, v in all_data.items()}
    keys = data['part'].astype('i8')*(1 << 32)+data['row']
    assert len(np.unique(keys)) == len(selected) == c.TARGETS.sum()
    np.testing.assert_array_equal(np.bincount(data['stratum'], minlength=48), allocation)
    assert np.all(allocation[population > 0] >= np.minimum(population[population > 0], 2))
    data['inclusion_probability'] = allocation[data['stratum']]/population[data['stratum']]
    source = c.ISO/f'x8_science_extension_20261102/time/{snap}'
    ratios = json.loads((source/'reservoir.json').read_text())['ratios']
    data['uniform'] = reader.uniform(data['actual'], ratios)
    data['nuclei'] = data['mass']*prod.MSUN_G*.76/prod.MH_G
    expected = json.loads((source/'summary.json').read_text())
    assert population.sum() == expected['active_count']
    prediction = c.parent.predict((data['actual'], data['slug'], data['uniform']))
    with h5py.File(folder/'sample.h5', 'x') as f:
        for k, v in data.items():f.create_dataset(k, data=v, compression='gzip')
        f.create_dataset('prediction', data=prediction, compression='gzip')
        f.create_dataset('population', data=population); f.create_dataset('allocation', data=allocation)
        f.attrs['manifest_sha256'] = manifest
    raw = np.stack((data['actual'], data['slug'], data['actual'], data['uniform'], data['uniform']), axis=1)
    with h5py.File(folder/'cases.h5', 'x') as f:
        f.create_dataset('raw', data=raw.reshape(-1, 7), compression='gzip')
        f.create_dataset('branch', data=np.tile(np.arange(5), len(selected)))
        f.create_dataset('particle_id', data=np.repeat(data['particle_id'], 5))
        f.attrs['manifest_sha256'] = manifest
    c.save(folder/'prepared.json', dict(complete=True, snapshot=snap, manifest_sha256=manifest,
        sample_sha256=c.sha(folder/'sample.h5'), cases_sha256=c.sha(folder/'cases.h5'),
        particles=len(selected), cases=5*len(selected), population=int(population.sum()),
        radius_allocation=allocation.reshape(16, 3).sum(axis=1), uniform_ratios=ratios,
        seed=c.SEED+snap*100, fitting=False, additional_time_accuracy_pending=True))
    print('Independent temporal sample frozen:', snap, len(selected), flush=True)


if __name__ == '__main__':main()
