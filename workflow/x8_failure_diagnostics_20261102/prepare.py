"""Freeze all remaining cases, their priority ordering, and fixed controls."""
import json

import h5py
import numpy as np

import diagnostics_common as c


def main():
    target = c.ROOT / 'cases.h5'
    assert not target.exists()
    audit = json.loads((c.PARENT / 'independent_audit.json').read_text())
    assert audit['passed'] and c.parent.verify() == audit['manifest_sha256']
    assert c.base.core.digest(c.PARENT / 'review.json') == audit['review_sha256']
    for path, sha in audit['output_hashes'].items():
        assert c.base.core.digest(path) == sha
    with np.load(c.PARENT / 'candidate_overlay.npz') as f:
        qualified, values = f['qualified'].copy(), f['values'].copy()
    missing = np.flatnonzero(~qualified.ravel())
    assert len(missing) == 2725
    n = len(qualified)
    with h5py.File(c.SOURCE / 'sample.h5') as f:
        data = {k: f['sample/' + k][:] for k in ('actual', 'slug', 'uniform', 'particle_id', 'nuclei', 'inclusion_probability', 'xyz')}
        proxy = f['sample/lum_per_H'][:]
    kernel = np.full((n, 9), np.nan)
    for part in range(4):
        with np.load(c.SOURCE / f'geometry/part{part:02d}.npz') as f:
            kernel[f['ids']] = f['kernel_weights']
    assert np.all(np.isfinite(kernel))
    importance = c.importance(proxy, data['nuclei'], data['inclusion_probability'], (kernel, c.base.region_masks(data['xyz']).astype(float)))
    rng = np.random.default_rng(c.SEED)
    controls = []
    for b in range(5):
        pool = np.flatnonzero(qualified[:, b])
        ranked = pool[np.argsort(-importance[pool, b], kind='stable')]
        chosen = np.r_[ranked[:16], rng.choice(ranked[16:], 16, replace=False)]
        controls.extend(chosen * 5 + b)
    controls = np.array(controls)
    assert len(np.unique(controls)) == 160 and not np.isin(controls, missing).any()
    base_ids = np.r_[missing, controls]
    kind = np.r_[np.zeros(len(missing), 'u1'), np.ones(len(controls), 'u1')]
    order = np.argsort(-importance.ravel()[base_ids], kind='stable')
    base_ids, kind = base_ids[order], kind[order]
    ids = np.repeat(base_ids, 2)
    protocol = np.tile(np.arange(2, dtype='u1'), len(base_ids))
    kind = np.repeat(kind, 2)
    partition = (np.arange(len(ids)) % 8).astype('u1')
    raw = np.empty((len(ids), 7))
    branch = ids % 5
    uid = ids // 5
    for b in range(5):
        key = 'slug' if b == 1 else 'uniform' if b >= 3 else 'actual'
        use = branch == b
        raw[use] = data[key][uid[use]]
    previous_error = np.full(n * 5, b'', 'S768')
    for part in range(8):
        with np.load(c.PARENT / f'output/part{part:02d}.npz') as f:
            previous_error[f['state_id']] = f['error']
    assert np.all(previous_error[missing] != b'')
    with h5py.File(target, 'x') as f:
        arrays = dict(state_id=ids, protocol=protocol, kind=kind, partition=partition, branch=branch,
                      raw=raw, particle_id=data['particle_id'][uid], priority=importance.ravel()[ids],
                      previous_values=values.reshape(n*5, 27)[ids], previous_error=previous_error[ids])
        for key, array in arrays.items():
            f.create_dataset(key, data=array, compression='gzip')
        f.attrs.update(complete=True, source_sample_sha256=c.base.core.digest(c.SOURCE / 'sample.h5'))
    concentration = {}
    for b, model in enumerate((0, 1, 3, 4, 5)):
        lost = np.flatnonzero(~qualified[:, b])
        weight = data['nuclei']/data['inclusion_probability']*kernel[:, 0]*proxy[:, model, 2]
        ranked = lost[np.argsort(-weight[lost])]
        fraction = np.cumsum(weight[ranked])/weight[lost].sum()
        concentration[c.base.BRANCHES[b]] = dict(unresolved=len(lost), n_for_90_percent=int(np.searchsorted(fraction, .9)+1),
              n_for_99_percent=int(np.searchsorted(fraction, .99)+1), top100_missing_fraction=float(fraction[99]))
    record = dict(total_cases=len(ids), unique_states=len(base_ids), target_states=len(missing), controls=len(controls),
                  protocols=c.PROTOCOLS, state_wall_seconds=c.WALL_SECONDS, alternative_mxstep=c.MXSTEP,
                  tEqGuess=None, maxTime_s=1e18, smallabd=1e-15, relative_tolerance=.001, seed=c.SEED,
                  partition_counts=np.bincount(partition, minlength=8), missing_C18O_concentration=concentration,
                  additional_particle_samples=0)
    assert len(ids) == 5770 and len(np.unique(np.column_stack((ids, protocol)), axis=0)) == len(ids)
    c.save(c.ROOT / 'case_design.json', record)
    print(json.dumps(c.base.clean(record), indent=2))


if __name__ == '__main__':
    main()
