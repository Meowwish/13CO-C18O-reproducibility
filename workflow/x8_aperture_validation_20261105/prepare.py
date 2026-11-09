"""Probability-sample apertures, then census every nonzero particle contribution."""
import json
import os

import h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix

import aperture_common as c
from kernel import make_patches, evaluate
from plot_galic_selection import lattice, sample


def main():
    assert int(os.environ.get('PBS_NCPUS', '0')) > 0
    manifest = c.verify()
    assert not (c.ROOT/'census.h5').exists()
    xy = lattice(); radius = np.hypot(*xy.T)
    cached = []
    for model, name in enumerate(c.MODELS):
        path = c.MAPS[model] if model < 2 else c.MAPS[2]
        prefix = '' if model < 2 else name+'_'
        with np.load(path) as f:
            cached.append(np.column_stack([sample(f[prefix+k], xy) for k in ('W13CO', 'WC18O')]))
    cached = np.stack(cached, axis=1)
    # Sort only frozen predictions, before consulting any direct labels.
    brightness = np.max(np.min(cached/np.array([.489, .432]), axis=2), axis=1)
    strata = np.full(len(xy), -1, 'i8'); selected = []; populations = []; allocations = []
    rng = np.random.default_rng(c.SEED)
    for radial in range(1, 15):
        ids = np.flatnonzero((radius >= radial)&(radius < radial+1))
        order = ids[np.argsort(brightness[ids], kind='stable')]
        for group, members in enumerate(np.array_split(order, 4)):
            h = len(populations); strata[members] = h
            nh = min(len(members), 2 if radial < 8 else 8)
            populations.append(len(members)); allocations.append(nh)
            selected.extend(rng.choice(members, nh, replace=False).tolist())
    selected = np.array(sorted(selected)); points = xy[selected]
    assert len(selected) == 280 and len(np.unique(selected)) == 280
    populations, allocations = np.array(populations), np.array(allocations)
    aperture_strata = strata[selected]; pi = allocations[aperture_strata]/populations[aperture_strata]
    patches, bounds, reach = make_patches(points)
    tree = cKDTree(points*1000)
    registry = json.loads((c.ISO/'x8_full_direct_575_20261028/registry.json').read_text())
    fields = ('actual', 'slug', 'mass', 'nuclei', 'xyz', 'h', 'row', 'particle_id', 'lum_per_H')
    target = c.ROOT/'census.h5'
    with h5py.File(str(target)+'.tmp', 'w') as out:
        out.attrs.update(complete=False, manifest_sha256=manifest, seed=c.SEED)
        for name, value in dict(xy=points, lattice_index=selected, stratum=aperture_strata,
                inclusion_probability=pi, population=populations, allocation=allocations,
                cached_line=cached[selected], patch_bounds=bounds, adjoint_patches=patches).items():
            out.create_dataset('apertures/'+name, data=value)
        for name, dtype in (('particle', 'i8'), ('aperture', 'i8'), ('weight', 'f8')):
            shape = (0, 2) if name == 'weight' else (0,)
            out.create_dataset('operator/'+name, shape=shape, maxshape=(None,)+shape[1:], dtype=dtype, chunks=True)
        count = 0; entries = 0; replay = np.zeros((280, 6, 2)); counts = np.zeros(280, 'i8')
        for part, entry in enumerate(registry['parts']):
            with h5py.File(entry['path']) as f:
                assert f.attrs['complete']
                for start in range(0, len(f['mass']), 100000):
                    stop = min(start+100000, len(f['mass']))
                    xyz, h = f['xyz'][start:stop], f['h'][start:stop]
                    candidates = tree.query_ball_point(xyz[:, :2], np.maximum(h, 40)+reach+1e-6, workers=8)
                    sizes = np.fromiter((len(x) for x in candidates), dtype='i8', count=len(candidates))
                    if sizes.sum() == 0:
                        continue
                    particle = np.repeat(np.arange(len(h)), sizes)
                    aperture = np.concatenate([x for x in candidates if len(x)]).astype('i8')
                    weight = evaluate(xyz, h, particle, aperture, patches, bounds)
                    nonzero = np.any(weight > 0, axis=1)
                    particle, aperture, weight = particle[nonzero], aperture[nonzero], weight[nonzero]
                    if not len(particle):
                        continue
                    unique, inverse = np.unique(particle, return_inverse=True)
                    block = {name: f[name][start:stop][unique] for name in fields}
                    block.update(part=np.full(len(unique), part), cache_row=unique+start)
                    for name, value in block.items():
                        if 'particles/'+name not in out:
                            out.create_dataset('particles/'+name, shape=(0,)+value.shape[1:],
                                maxshape=(None,)+value.shape[1:], dtype=value.dtype, chunks=True)
                        ds = out['particles/'+name]; ds.resize(count+len(unique), axis=0); ds[count:] = value
                    for name, value in dict(particle=inverse+count, aperture=aperture, weight=weight).items():
                        ds = out['operator/'+name]; ds.resize(entries+len(aperture), axis=0); ds[entries:] = value
                    for model in range(6):
                        op = int(model >= 2)
                        matrix = coo_matrix((weight[:, op], (aperture, inverse)), shape=(280, len(unique))).tocsr()
                        replay[:, model] += matrix@(block['nuclei'][:, None]*block['lum_per_H'][:, model, 1:3])
                    counts += np.bincount(aperture, minlength=280)
                    count += len(unique); entries += len(aperture)
            out.flush()
            print('census cache', part, 'unique particles', count, 'contributions', entries, flush=True)
        from process_snapshot563_lookup_v11 import luminosity_to_w
        config = json.loads((c.ISO/'highres_snapshot575_temporal/config.json').read_text())
        frequencies = config['projection']['line_frequencies_Hz']
        factors = np.array([luminosity_to_w(1., frequencies[k]) for k in ('13CO', 'C18O')])
        replay *= factors
        # A mandatory, full-input replay gate protects exact support selection,
        # projection conventions, frequencies and both map-edge operators.
        np.testing.assert_allclose(replay, cached[selected], rtol=3e-7, atol=1e-20)
        out.create_dataset('apertures/replayed_line', data=replay)
        out.create_dataset('apertures/contributor_count', data=counts)
        out.create_dataset('conversion_line', data=factors)
        # Molecular outputs permit selected molecule/emission checks as well.
        out.create_dataset('prediction', shape=(count, 6, 6), dtype='f8', chunks=True)
        for start in range(0, count, 100000):
            stop = min(start+100000, count)
            actual, slug = [out['particles/'+k][start:stop] for k in ('actual', 'slug')]
            uniform = c.base.uniform(actual, registry['uniform_ratios'])
            prediction = c.predict((actual, slug, uniform))
            np.testing.assert_allclose(prediction[:, :, 3:], out['particles/lum_per_H'][start:stop], rtol=2e-12, atol=1e-60)
            out['prediction'][start:stop] = prediction
        # Reference reuse is checked only after the independent aperture draw.
        reuse = np.full((count, 5, 27), np.nan); qualified = np.zeros((count, 5), bool)
        keys = out['particles/part'][:].astype('i8')*(1 << 32)+out['particles/cache_row'][:]
        assert len(np.unique(keys)) == count and np.all(np.diff(keys) > 0)
        for old_sample, old_direct in [
            (c.ISO/'x8_bounded_reference_20261101/sample.h5', c.ISO/'x8_failure_diagnostics_20261102/diagnostic_candidates.npz'),
            (c.ISO/'x8_outer_validation_20261105/sample.h5', None)]:
            with h5py.File(old_sample) as f:
                old_keys = f['sample/part'][:].astype('i8')*(1 << 32)+f['sample/cache_row'][:]
                loc = np.searchsorted(keys, old_keys); hit = loc < count; hit[hit] &= keys[loc[hit]] == old_keys[hit]
                old_indices = np.flatnonzero(hit); new_indices = loc[hit]
                np.testing.assert_array_equal(out['particles/particle_id'][:][new_indices], f['sample/particle_id'][:][old_indices])
                for name in ('actual', 'slug'):
                    np.testing.assert_array_equal(out['particles/'+name][:][new_indices], f['sample/'+name][:][old_indices])
                np.testing.assert_array_equal(c.base.uniform(out['particles/actual'][:][new_indices], registry['uniform_ratios']),
                    f['sample/uniform'][:][old_indices])
                nold = len(old_keys)
            if old_direct is not None:
                with np.load(old_direct) as f:
                    ov, oq = f['values'], f['qualified']
            else:
                ov = np.full((nold*5, 27), np.nan); oq = np.zeros(nold*5, bool); seen = oq.copy()
                for p in range(8):
                    with np.load(c.ISO/f'x8_outer_validation_20261105/direct/part{p:02d}.npz') as f:
                        ids = f['task_ids']; assert bool(f['complete']) and not seen[ids].any()
                        ov[ids], oq[ids], seen[ids] = f['values'], f['qualified'], True
                assert seen.all(); ov, oq = ov.reshape(nold, 5, 27), oq.reshape(nold, 5)
            for b in range(5):
                good = oq[old_indices, b]&(~qualified[new_indices, b])
                reuse[new_indices[good], b] = ov[old_indices[good], b]
                qualified[new_indices[good], b] = True
        out.create_dataset('reuse_values', data=reuse, compression='gzip')
        out.create_dataset('reuse_qualified', data=qualified, compression='gzip')
        out.attrs['complete'] = True
    os.replace(str(target)+'.tmp', target)
    c.save(c.ROOT/'prepared.json', dict(complete=True, manifest_sha256=manifest,
        census_sha256=c.sha(target), apertures=280, outer_apertures=224, inner_apertures=56,
        unique_particles=count, logical_states=count*5, reused_qualified_states=int(qualified.sum()),
        contributor_count=counts, full_input_replay_passed=True,
        scope='Every positive-support active particle is retained; no brightness truncation.'))
    print('Census prepared:', count, 'particles;', count*5, 'logical states.', flush=True)


if __name__ == '__main__':
    main()
