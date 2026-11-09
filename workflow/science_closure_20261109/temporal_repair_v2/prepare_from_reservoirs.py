"""Retain census layers below their planned target; reuse frozen priorities."""
from pathlib import Path
import json
import os
import sys

import h5py
import numpy as np

REPAIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPAIR.parent))
import closure_common as c
import temporal_prepare as original


def main():
    assert int(os.environ.get('PBS_NCPUS', '0')) > 0
    manifest = c.verify(); snap = int(os.environ['SNAPSHOT']); assert snap in (563, 567)
    repair = json.loads((REPAIR/'manifest.json').read_text())
    assert repair['original_manifest_sha256'] == manifest
    for path, digest in repair['hashes'].items():assert c.sha(path) == digest, path
    folder = c.ROOT/f'temporal{snap}'
    assert not (folder/'sample.h5').exists() and not (folder/'cases.h5').exists()
    blocks = []; population = np.zeros(48, 'i8')
    for part in range(8):
        path = folder/f'reservoir{part:02d}.npz'
        assert c.sha(path) == repair['reservoir_sha256'][str(path)]
        with np.load(path) as f:
            population += f['population']
            blocks.append({k:f[k].copy() for k in f.files if k != 'population'})
    all_data = {k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    allocation = np.zeros(48, 'i8'); selected = []
    for radial in range(16):
        s = slice(radial*3, radial*3+3); N = population[s]
        allocation[s] = c.parent.base.allocation(N, min(int(c.TARGETS[radial]), int(N.sum())))
        for h in range(radial*3, radial*3+3):
            ids = np.flatnonzero(all_data['stratum'] == h)
            selected.extend(ids[np.argsort(all_data['priority'][ids])[:allocation[h]]])
    data = {k:v[np.asarray(selected)] for k, v in all_data.items()}
    expected = int(np.minimum(population.reshape(16, 3).sum(1), c.TARGETS).sum())
    keys = data['part'].astype('i8')*(1 << 32)+data['row']
    assert len(np.unique(keys)) == len(selected) == expected
    np.testing.assert_array_equal(np.bincount(data['stratum'], minlength=48), allocation)
    assert np.all((population == 0) | (allocation >= np.minimum(population, 2)))
    data['inclusion_probability'] = allocation[data['stratum']]/population[data['stratum']]
    source = c.ISO/f'x8_science_extension_20261102/time/{snap}'
    ratios = json.loads((source/'reservoir.json').read_text())['ratios']
    data['uniform'] = original.reader.uniform(data['actual'], ratios)
    data['nuclei'] = data['mass']*original.prod.MSUN_G*.76/original.prod.MH_G
    assert population.sum() == json.loads((source/'summary.json').read_text())['active_count']
    prediction = c.parent.predict((data['actual'], data['slug'], data['uniform']))
    repair_sha = c.sha(REPAIR/'manifest.json')
    with h5py.File(folder/'sample.h5', 'x') as f:
        for k, v in data.items():f.create_dataset(k, data=v, compression='gzip')
        f.create_dataset('prediction', data=prediction, compression='gzip')
        f.create_dataset('population', data=population);f.create_dataset('allocation', data=allocation)
        f.attrs.update(manifest_sha256=manifest, preparation_repair_sha256=repair_sha)
    raw = np.stack((data['actual'], data['slug'], data['actual'], data['uniform'], data['uniform']), axis=1)
    with h5py.File(folder/'cases.h5', 'x') as f:
        f.create_dataset('raw', data=raw.reshape(-1, 7), compression='gzip')
        f.create_dataset('branch', data=np.tile(np.arange(5), len(selected)))
        f.create_dataset('particle_id', data=np.repeat(data['particle_id'], 5))
        f.attrs.update(manifest_sha256=manifest, preparation_repair_sha256=repair_sha)
    c.save(folder/'prepared.json', dict(complete=True, snapshot=snap, manifest_sha256=manifest,
        preparation_repair_sha256=repair_sha, sample_sha256=c.sha(folder/'sample.h5'),
        cases_sha256=c.sha(folder/'cases.h5'), particles=len(selected), cases=5*len(selected),
        population=int(population.sum()), radius_allocation=allocation.reshape(16, 3).sum(1),
        uniform_ratios=ratios, seed=c.SEED+snap*100, fitting=False,
        actual_count_below_maximum_because_finite_layer_is_censused=True))
    print('Reused frozen random priorities; finite outer layer censused:', snap, len(selected), flush=True)


if __name__ == '__main__':main()
