"""Freeze known direct states and probability-sample unresolved contributors."""
import json
import os

import h5py
import numpy as np
from scipy.sparse import coo_matrix

import precision_common as c
from precision_design import allocate
from precision_solve import read_rows


def main():
    assert int(os.environ.get('PBS_NCPUS', '0')) > 0
    manifest = c.verify()
    assert not (c.ROOT/'sample.h5').exists()
    parent_manifest, pr = c.parent.ready()
    n = pr['unique_particles']; snapshots = json.loads((c.ROOT/'manifest.json').read_text())['checkpoint_snapshots']
    status = np.zeros(n*5, 'i1'); qualified = np.zeros(n*5, bool)
    values = np.full((n*5, 6), np.nan); seen = np.zeros(n*5, bool)
    source_hashes = {}
    for part in range(32):
        path = c.PARENT/f'direct/part{part:02d}.npz'
        source_hashes[str(path)] = c.sha(path)
        with np.load(path) as f:
            ids = f['task_ids']; assert not seen[ids].any()
            metadata = json.loads(str(f['metadata']))
            assert metadata['census_sha256'] == pr['census_sha256']
            np.testing.assert_array_equal(f['qualified'], (f['status'] == 1)&np.all(f['final_returns'] == 1, axis=1))
            status[ids], qualified[ids], values[ids], seen[ids] = f['status'], f['qualified'], f['values'][:, 3:9], True
    assert seen.all()
    status, qualified, values = status.reshape(n, 5), qualified.reshape(n, 5), values.reshape(n, 5, 6)
    assert np.isfinite(values[qualified]).all()
    attempted = status != 0
    controls = [0, 2, 3, 4]
    known_control = attempted[:, controls].all(axis=1)|np.any(attempted[:, controls]&(~qualified[:, controls]), axis=1)
    known = np.column_stack((attempted[:, 0], attempted[:, 1], *[known_control]*4))
    with h5py.File(c.PARENT/'census.h5') as f:
        prediction = f['prediction'][:]
        mass, nuclei = f['particles/mass'][:], f['particles/nuclei'][:]
        density = np.digitize(np.log10(f['particles/actual'][:, 0]), [1, 3])
        part_ids, cache_rows = f['particles/part'][:], f['particles/cache_row'][:]
        ap = {k: f['apertures/'+k][:] for k in ('xy', 'lattice_index', 'stratum', 'inclusion_probability', 'population', 'allocation', 'cached_line')}
        particle, aperture, kernel = [f['operator/'+k][:] for k in ('particle', 'aperture', 'weight')]
        factors = f['conversion_line'][:]
    matrices = [coo_matrix((kernel[:, op], (aperture, particle)), shape=(280, n)).tocsr() for op in range(2)]
    baseline = np.zeros((280, 6, 8)); full_proxy = np.zeros((280, 6, 4))
    for model in range(6):
        branch = c.MODEL_BRANCH[model]; matrix = matrices[int(model >= 2)]
        k = known[:, model]
        q = qualified[:, branch] if model < 2 else qualified[:, controls].all(axis=1)
        dline = nuclei[:, None]*np.where((k&q)[:, None], values[:, branch, 4:6], 0.)*factors
        dmolecule = mass[:, None]*np.where((k&q)[:, None], values[:, branch, 1:3], 0.)
        pline = nuclei[:, None]*prediction[:, model, 4:6]*factors
        pmolecule = mass[:, None]*prediction[:, model, 1:3]
        uline, umolecule = matrix@((~k)[:, None]*pline), matrix@((~k)[:, None]*pmolecule)
        baseline[:, model, :2] = matrix@dline+uline
        baseline[:, model, 2:4] = matrix@((k&q)[:, None]*pline)+uline
        baseline[:, model, 4:6] = matrix@dmolecule+umolecule
        baseline[:, model, 6:8] = matrix@((k&q)[:, None]*pmolecule)+umolecule
        full_proxy[:, model, :2], full_proxy[:, model, 2:] = matrix@pline, matrix@pmolecule
    np.testing.assert_allclose(full_proxy[:, :, :2], ap['cached_line'], rtol=3e-7, atol=1e-20)
    rng = np.random.default_rng(c.SEED)
    order = np.argsort(aperture, kind='stable'); boundaries = np.searchsorted(aperture[order], np.arange(281))
    records = []; populations = np.zeros((280, 16), 'i8'); allocations = populations.copy()
    unknown_counts = np.zeros(280, 'i8')
    for a in range(280):
        entries = order[boundaries[a]:boundaries[a+1]]
        ids, wk = particle[entries], kernel[entries]
        assert len(np.unique(ids)) == len(ids)
        take = (~known[ids]).any(axis=1)
        ids, wk = ids[take], wk[take]; unknown_counts[a] = len(ids)
        if not len(ids):
            continue
        importance = np.zeros(len(ids))
        for model in range(6):
            op = int(model >= 2)
            light = wk[:, op, None]*nuclei[ids, None]*prediction[ids, model, 4:6]*factors
            molecules = wk[:, op, None]*mass[ids, None]*prediction[ids, model, 1:3]
            contributions = np.column_stack((light, molecules))
            scale = np.maximum(full_proxy[a, model], 1e-300)
            imp = np.max(contributions/scale, axis=1)
            importance = np.maximum(importance, np.where(known[ids, model], 0., imp))
        # Deterministic bright contributors are an exact census stratum;
        # every remaining particle retains positive SRS inclusion probability.
        ranked = np.argsort(-importance, kind='stable')
        top = ranked[:min(c.TOP, len(ids))]; tail = ranked[len(top):]
        populations[a, 15] = allocations[a, 15] = len(top)
        chosen = [top]; labels = [np.full(len(top), 15, 'i8')]
        if len(tail):
            bins = np.digitize(np.log10(np.maximum(importance[tail], 1e-300)), [-10, -8, -6, -4])
            strata = density[ids[tail]]*5+bins
            N = np.bincount(strata, minlength=15)
            proxy_sd = np.sqrt(N*np.bincount(strata, weights=importance[tail]**2, minlength=15))
            nh = allocate(N, proxy_sd, min(c.TAIL, len(tail)))
            populations[a, :15], allocations[a, :15] = N, nh
            for h in np.flatnonzero(N):
                members = tail[strata == h]
                pick = rng.choice(members, int(nh[h]), replace=False)
                chosen.append(pick); labels.append(np.full(len(pick), h, 'i8'))
        pick, sh = np.concatenate(chosen), np.concatenate(labels)
        assert len(np.unique(pick)) == len(pick)
        assert np.all((populations[a] == 0)|(allocations[a] >= np.minimum(populations[a], 2)))
        records.append(dict(census_row=ids[pick], aperture=np.full(len(pick), a), stratum=sh,
            pi=allocations[a, sh]/populations[a, sh], kernel=wk[pick], unknown=~known[ids[pick]]))
    assert records
    records = {k: np.concatenate([r[k] for r in records]) for k in records[0]}
    unique, inverse = np.unique(records['census_row'], return_inverse=True)
    records['particle'] = inverse
    assert len(unique) <= 280*(c.TOP+c.TAIL)
    tasks = np.flatnonzero((status[unique] == 0).ravel())
    assert len(tasks) <= 5*280*(c.TOP+c.TAIL)
    with h5py.File(c.PARENT/'census.h5') as source, h5py.File(str(c.ROOT/'sample.h5')+'.tmp', 'w') as out:
        out.attrs.update(complete=False, manifest_sha256=manifest, parent_manifest_sha256=parent_manifest)
        for k, v in ap.items(): out.create_dataset('apertures/'+k, data=v)
        for k, v in records.items(): out.create_dataset('records/'+k, data=v)
        for k, v in dict(population=populations, allocation=allocations, baseline=baseline, full_proxy=full_proxy,
                unknown_counts=unknown_counts, conversion_line=factors, tasks=tasks).items(): out.create_dataset(k, data=v)
        for k in ('actual', 'slug', 'mass', 'nuclei', 'particle_id'):
            out.create_dataset('sample/'+k, data=read_rows(source['particles/'+k], unique))
        registry = json.loads((c.ISO/'x8_full_direct_575_20261028/registry.json').read_text())
        out.create_dataset('sample/uniform', data=c.base.uniform(out['sample/actual'][:], registry['uniform_ratios']))
        for k, v in dict(census_row=unique, part=part_ids[unique], cache_row=cache_rows[unique],
                prediction=prediction[unique], status=status[unique], qualified=qualified[unique], values=values[unique]).items():
            out.create_dataset('sample/'+k, data=v, compression='gzip')
        out.attrs['complete'] = True
    os.replace(str(c.ROOT/'sample.h5')+'.tmp', c.ROOT/'sample.h5')
    c.save(c.ROOT/'prepared.json', dict(complete=True, manifest_sha256=manifest, sample_sha256=c.sha(c.ROOT/'sample.h5'),
        parent_census_sha256=pr['census_sha256'], existing_qualified_states=int(qualified.sum()),
        existing_attempted_states=int(attempted.sum()), remaining_original_states=int((~attempted).sum()),
        particle_records=len(inverse), distinct_reference_particles=len(unique), new_states=len(tasks),
        maximum_new_states=5*280*(c.TOP+c.TAIL), full_proxy_replay_passed=True,
        estimated_compute_hours_at_nine_nodes=len(tasks)/(11500*9),
        parent_checkpoint_hashes=source_hashes,
        scope='Exact existing-state contributions plus positive-probability residual sampling of unresolved particle contributions. '
            'Reference precision must be reviewed; no complete particle-census claim.'))
    print('Prepared precision reference:', len(unique), 'distinct particles;', len(tasks), 'new states.', flush=True)


if __name__ == '__main__':
    main()
