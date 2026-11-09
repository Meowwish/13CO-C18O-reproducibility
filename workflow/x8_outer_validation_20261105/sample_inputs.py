"""Two full-frame streaming passes, with 4096 draws per outer annulus."""
import json
import os
from pathlib import Path

import h5py
import numpy as np

import outer_common as c

FIELDS = ('actual', 'slug', 'mass', 'xyz', 'h', 'row', 'particle_id',
          'nuclei', 'lum_per_H', 'covered')


def chunks(path):
    with h5py.File(path) as f:
        assert f.attrs['complete']
        for start in range(0, len(f['mass']), 100000):
            d = {k: f[k][start:start+100000] for k in FIELDS}
            d['cache_row'] = np.arange(start, start+len(d['mass']))
            yield d


def main():
    assert int(os.environ.get('PBS_NCPUS', '0')) > 0
    sha = c.verify()
    target = c.ROOT/'sample.h5'
    assert not target.exists()
    plan = c.base.core.load_plan()
    paths = [Path(x['path']) for x in plan['parts']]
    for p, entry in zip(paths, plan['parts']):
        s = p.stat()
        assert s.st_size == entry['size'] and s.st_mtime_ns == entry['mtime_ns']
    with h5py.File(c.SOURCE/'sample.h5') as f:
        totals = f['predicted_particle_totals'][0]
    ratios = plan['uniform_ratios']
    population = np.zeros(16*c.STRATA_PER_RADIUS, 'i8')
    for part, path in enumerate(paths):
        for d in chunks(path):
            population += np.bincount(c.strata(d, ratios, totals), minlength=len(population))
        print('counted', part, flush=True)
    assert population.sum() == plan['active_count']
    allocation = c.allocate(population)
    rng = np.random.default_rng(c.SEED)
    ranks = {h: np.sort(rng.choice(int(population[h]), int(allocation[h]), replace=False))
             for h in np.flatnonzero(population)}
    offset = np.zeros_like(population)
    blocks = []
    for part, path in enumerate(paths):
        for d in chunks(path):
            s = c.strata(d, ratios, totals)
            for h in np.unique(s):
                indices = np.flatnonzero(s == h)
                lo, hi = offset[h], offset[h]+len(indices)
                a, b = np.searchsorted(ranks[h], [lo, hi])
                pick = indices[ranks[h][a:b]-lo]
                if len(pick):
                    blocks.append({k: v[pick] for k, v in d.items()} |
                                  dict(part=np.full(len(pick), part), stratum=np.full(len(pick), h)))
                offset[h] = hi
        print('sampled', part, flush=True)
    np.testing.assert_array_equal(offset, population)
    selected = {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    order = np.lexsort((selected['cache_row'], selected['part']))
    selected = {k: v[order] for k, v in selected.items()}
    n = len(order)
    np.testing.assert_array_equal(np.bincount(selected['stratum'], minlength=len(population)), allocation)
    keys = selected['part'].astype('i8')*(1 << 32)+selected['cache_row']
    assert len(np.unique(keys)) == n and n <= c.RADIAL_TARGETS.sum()
    selected['inclusion_probability'] = allocation[selected['stratum']]/population[selected['stratum']]
    selected['uniform'] = c.base.uniform(selected['actual'], ratios)
    # The independent draw is now fixed; only now consult existing direct labels.
    values = np.full((n, 5, 27), np.nan)
    qualified = np.zeros((n, 5), bool)
    with h5py.File(c.SOURCE/'sample.h5') as f:
        old_keys = f['sample/part'][:].astype('i8')*(1 << 32)+f['sample/cache_row'][:]
        old_order = np.argsort(old_keys)
        old_keys = old_keys[old_order]
        loc = np.searchsorted(old_keys, keys)
        match = loc < len(old_keys)
        match[match] &= old_keys[loc[match]] == keys[match]
        new_idx = np.flatnonzero(match)
        old_idx = old_order[loc[match]]
        for k in ('actual', 'slug', 'uniform', 'particle_id', 'row'):
            np.testing.assert_array_equal(selected[k][new_idx], f['sample/'+k][:][old_idx])
    with np.load(c.PARENT/'diagnostic_candidates.npz') as f:
        values[new_idx] = f['values'][old_idx]
        qualified[new_idx] = f['qualified'][old_idx]
    values[~qualified] = np.nan
    tmp = Path(str(target)+'.tmp')
    with h5py.File(tmp, 'w') as f:
        f.attrs.update(complete=False, manifest_sha256=sha, seed=c.SEED,
                       independent_outer_focused_design=True, validation_only=True)
        for k, v in selected.items():
            f.create_dataset('sample/'+k, data=v, compression='gzip')
        f.create_dataset('stratum_population', data=population)
        f.create_dataset('stratum_sample', data=allocation)
        f.create_dataset('reuse_values', data=values, compression='gzip')
        f.create_dataset('reuse_qualified', data=qualified, compression='gzip')
        f.attrs['complete'] = True
    os.replace(tmp, target)
    c.save(c.ROOT/'sample_ready.json', dict(complete=True, manifest_sha256=sha,
        sample_sha256=c.digest(target), particle_count=n, physical_states=5*n,
        reused_qualified_states=int(qualified.sum()), radial_population=population.reshape(16, -1).sum(1),
        radial_sample=allocation.reshape(16, -1).sum(1), stratum_population=population,
        stratum_sample=allocation, note='Independent SRSWOR; do not naively pool with the old sample.'))
    print('Frozen sample', n, 'states', 5*n, 'reused', qualified.sum(), flush=True)


if __name__ == '__main__':
    main()
