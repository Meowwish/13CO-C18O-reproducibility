"""Retry every unresolved state in the two independent particle cohorts."""
import json
import os

import h5py
import numpy as np

import closure_common as c


def cohorts():
    for name, source in [('global', c.ISO/'x8_bounded_reference_20261101'),
                         ('outer', c.ISO/'x8_outer_validation_20261105')]:
        with h5py.File(source/'sample.h5') as f:
            d = {k:f['sample/'+k][:] for k in ('actual', 'slug', 'uniform', 'particle_id',
                'nuclei', 'mass', 'stratum', 'inclusion_probability')}
            d['population'] = f['stratum_population'][:]
        n = len(d['mass'])
        if name == 'global':
            with np.load(c.ISO/'x8_failure_diagnostics_20261102/diagnostic_candidates.npz') as f:
                values, qualified = f['values'].copy(), f['qualified'].copy()
        else:
            values = np.full((n, 5, 27), np.nan); qualified = np.zeros((n, 5), bool)
            for part in range(8):
                with np.load(source/f'direct/part{part:02d}.npz') as f:
                    ids = f['task_ids']
                    values.reshape(-1, 27)[ids] = f['values']
                    qualified.ravel()[ids] = f['qualified']
        yield name, d, values, qualified


def main():
    assert int(os.environ.get('PBS_NCPUS', '0')) > 0
    manifest = c.verify(); folder = c.ROOT/'recovery'; folder.mkdir(exist_ok=True)
    assert not (folder/'cases.h5').exists()
    cases = []
    rng = np.random.default_rng(c.SEED)
    for cohort, d, values, q in cohorts():
        chosen = [(int(i), 0) for i in np.flatnonzero(~q.ravel())]
        for branch in range(5):
            pool = np.flatnonzero(q[:, branch])
            chosen += [(int(5*i+branch), 1) for i in rng.choice(pool, 32, replace=False)]
        for state, control in chosen:
            i, b = divmod(state, 5)
            key = 'slug' if b == 1 else 'uniform' if b >= 3 else 'actual'
            cases.append(dict(cohort=cohort, state_id=state, branch=b, control=control,
                raw=d[key][i], particle_id=d['particle_id'][i], primary=values[i, b]))
    with h5py.File(folder/'cases.h5', 'x') as f:
        for key in cases[0]:
            data = [x[key] for x in cases]
            if key == 'cohort':data = np.asarray(data, dtype='S8')
            f.create_dataset(key, data=data, compression='gzip')
        f.attrs['manifest_sha256'] = manifest
    c.save(folder/'prepared.json', dict(complete=True, manifest_sha256=manifest,
        cases_sha256=c.sha(folder/'cases.h5'), cases=len(cases),
        unresolved=sum(not x['control'] for x in cases), controls=sum(x['control'] for x in cases),
        protocol=dict(tEqGuess=None, smallabd=1e-15, tol=.001, mxstep=50000,
                      wall_seconds=1800, maxTime_s=1e18),
        scope='All unresolved global and outer reference states, plus 32 matched successful controls per branch/cohort.'))
    print('Recovery cases frozen:', len(cases), flush=True)


if __name__ == '__main__':main()
