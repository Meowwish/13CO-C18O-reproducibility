"""Outer molecule / emission checks using existing independent direct labels."""
import hashlib
import json
import os
from pathlib import Path
import sys

import h5py
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
ISO = ROOT.parent
sys.path.insert(0, str(ISO/'x8_claim_checks_20261105_v3'))
import claim_common as c
import outer_common as outer
from adaptive_local_surrogate import (AdaptiveLocalActualAbundanceSurrogate, OUTPUT_NAMES,
    transform_outputs, raw_inputs_to_features, inverse_transform_outputs)


def main():
    assert int(os.environ.get('PBS_NCPUS', '0')) > 0
    assert not (ROOT/'review.json').exists()
    manifest = json.loads((ROOT/'manifest.json').read_text())
    for path, sha in manifest['hashes'].items():
        assert c.sha(path) == sha, path
    outer_sha, ready = outer.ready()
    assert outer_sha == manifest['outer_manifest_sha256']
    with h5py.File(outer.ROOT/'sample.h5') as f:
        raw = [f['sample/'+key][:] for key in ('actual', 'slug', 'uniform')]
        mass, nuclei, pi, strata = [f['sample/'+key][:] for key in
            ('mass', 'nuclei', 'inclusion_probability', 'stratum')]
        population, allocation = f['stratum_population'][:], f['stratum_sample'][:]
        previous = f['sample/lum_per_H'][:]
    np.testing.assert_array_equal(pi, allocation[strata]/population[strata])
    design = c.sfr.groups(strata, population)
    base = AdaptiveLocalActualAbundanceSurrogate(ISO/'chemTable_temporal_550_575/final_test_600/frozen_surrogate.joblib', require_validated=False)
    bundle = joblib.load(ISO/'isotope_fractionation_controls/intervention_models/residual_leaves15.joblib')
    assert bundle['base_sha256'] == c.sha(ISO/'chemTable_temporal_550_575/final_test_600/frozen_surrogate.joblib')
    prediction = []
    for x in raw[:2]:
        v, _ = base.query(x, chunk_size=100000)
        prediction.append(np.column_stack([v[name] for name in OUTPUT_NAMES]))
    for x in (raw[0], raw[2]):
        v, _ = base.query(x, chunk_size=100000)
        bt = transform_outputs(x, np.column_stack([v[name] for name in OUTPUT_NAMES]))
        augmented = np.column_stack((raw_inputs_to_features(x), bt))
        for physics in ('on', 'off'):
            correction = np.column_stack([model.predict(augmented) for model in bundle['models'][physics]])
            prediction.append(inverse_transform_outputs(x, bt+correction, bundle['target_clips'][physics]))
    prediction = np.stack(prediction, axis=1)
    np.testing.assert_allclose(prediction[:, :, 3:], previous, rtol=2e-12, atol=1e-60)
    n = len(mass)
    values = np.full((n*5, 27), np.nan); qualified = np.zeros(n*5, bool); seen = qualified.copy()
    for part in range(8):
        with np.load(outer.ROOT/f'direct/part{part:02d}.npz') as f:
            ids = f['task_ids']; assert not seen[ids].any() and bool(f['complete'])
            assert json.loads(str(f['metadata']))['sample_sha256'] == ready['sample_sha256']
            values[ids], qualified[ids], seen[ids] = f['values'], f['qualified'], True
    assert seen.all()
    values, qualified = values.reshape(n, 5, 27), qualified.reshape(n, 5)
    kernels = np.full((n, 4, 19), np.nan); seen = np.zeros(n, bool)
    for part in range(4):
        with np.load(outer.ROOT/f'geometry/part{part:02d}.npz') as f:
            ids = f['ids']; assert not seen[ids].any() and bool(f['complete'])
            kernels[ids], seen[ids] = f['kernel'], True
    assert seen.all() and np.isfinite(kernels).all()
    records = []; contrasts = []; checked = 0; max_delta = 0.
    for op_group in range(2):
        for model, name in enumerate(c.MODELS):
            b = c.MODEL_BRANCH[model]; q = qualified[:, b]
            k = kernels[:, op_group*2+int(model >= 2)]
            line = c.estimate(values[:, b, 7:9], prediction[:, model, 4:6], q, k, nuclei/pi)
            molecule = c.estimate(values[:, b, 4:6], prediction[:, model, 1:3], q, k, mass/pi)
            effective = {key: line[key]-molecule[key] for key in ('direct', 'proxy', 'direct_score', 'proxy_score')}
            for label, m in (('line_ratio', line), ('molecule_number_ratio', molecule), ('emission_per_molecule_ratio', effective)):
                for j in range(8, 15):
                    direct = c.sfr.interval(m['direct'][j], m['direct_score'][:, j], design)
                    error = c.sfr.interval(m['direct'][j]-m['proxy'][j], m['direct_score'][:, j]-m['proxy_score'][:, j], design)
                    ci = error['interval_95']
                    records.append(dict(geometry=('native', 'lattice')[op_group], model=name, observable=label,
                        region=c.REGIONS[j], direct_log_ratio=direct, matched_surrogate_log_ratio=m['proxy'][j],
                        direct_minus_surrogate=error, interval_within_0p05_dex=ci is not None and ci[0] >= -.05 and ci[1] <= .05,
                        missing_predicted_line_fraction=line['missing_predicted_positive_fraction'][j],
                        missing_predicted_molecular_fraction=molecule['missing_predicted_positive_fraction'][j]))
                    # Independently reconstruct sums and scalar variance rather
                    # than reusing the main estimator for the audit.
                    pairs = []
                    for lo, hi, pcols, wt in ((7, 9, slice(4, 6), nuclei/pi), (4, 6, slice(1, 3), mass/pi)):
                        w = k[:, j]*wt
                        dv = np.where(q[:, None], values[:, b, lo:hi], 0.)
                        pv = np.where(q[:, None], prediction[:, model, pcols], 0.)
                        ds = np.sum(w[:, None]*dv, axis=0); ps = np.sum(w[:, None]*pv, axis=0)
                        d = np.log10(ds[0])-np.log10(ds[1]); p = np.log10(ps[0])-np.log10(ps[1])
                        sd = w*(dv[:, 0]/ds[0]-dv[:, 1]/ds[1])/np.log(10)
                        sp = w*(pv[:, 0]/ps[0]-pv[:, 1]/ps[1])/np.log(10)
                        pairs.append((d, p, sd, sp))
                    independent = pairs[0] if label == 'line_ratio' else pairs[1] if label == 'molecule_number_ratio' else tuple(a-bb for a, bb in zip(*pairs))
                    d, p, sd, sp = independent
                    for record, point, score in ((direct, d, sd), (error, d-p, sd-sp)):
                        variance = 0.
                        for ids, N in design:
                            nh = len(ids)
                            if nh < N:
                                centred = score[ids]-np.mean(score[ids])
                                variance += nh*(1-nh/N)*np.dot(centred, centred)/(nh-1)
                        se = float(np.sqrt(max(variance, 0.)))
                        max_delta = max(max_delta, abs(record['point']-point), abs(record['standard_error']-se))
                        np.testing.assert_allclose([record['point'], record['standard_error']], [point, se], atol=3e-12, rtol=0)
                        checked += 1
                for title, a in [('endpoint_14_15_minus_8_9', np.eye(19)[14]-np.eye(19)[8]),
                        ('outer_slope_dex_per_kpc', np.r_[np.zeros(8), (np.arange(8.5, 15)-11.5)/28, np.zeros(4)])]:
                    result, _, _ = c.contrast(m, a, design)
                    contrasts.append(dict(geometry=('native', 'lattice')[op_group], model=name, observable=label, contrast=title, **result))
    c.save(ROOT/'review.json', dict(complete=True, manifest_sha256=c.sha(ROOT/'manifest.json'),
        records=records, contrasts=contrasts, particles=n, new_chemical_solves=0, model_training_performed=False,
        scope='Uncut ratios, matched successful states; sampling uncertainty only. No selected-pixel or failed-state-light validation.'))
    c.save(ROOT/'independent_audit.json', dict(passed=True, checked_intervals=checked, maximum_difference=max_delta,
        luminosity_replay_passed=True, review_sha256=c.sha(ROOT/'review.json')))
    c.base.atomic_npz(ROOT/'predictions.npz', predicted=prediction)
    print('Outer molecular check complete:', len(records), 'records;', checked, 'audited intervals.', flush=True)


if __name__ == '__main__':
    main()
