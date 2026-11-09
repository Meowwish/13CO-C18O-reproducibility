"""Independent scalar reconstruction of the completed shielding experiments."""
import hashlib
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
ISO = ROOT.parent
SHIELD = ISO/'x8_shielding_sensitivity_20261105'
sys.path.append(str(SHIELD))
import shield_common as source


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    assert int(os.environ.get('PBS_NCPUS', '0')) > 0
    manifest, prepared = source.ready()
    review = json.loads((SHIELD/'review.json').read_text())
    assert review['complete'] and review['manifest_sha256'] == manifest
    n = prepared['particles']; values = np.full((n*20, 27), np.nan); q = np.zeros(n*20, bool); seen = q.copy()
    for part in range(16):
        path = SHIELD/f'direct/part{part:02d}.npz'
        assert sha(path) == review['output_hashes'][str(path)]
        with np.load(path) as f:
            ids = f['task_ids']; assert not seen[ids].any() and bool(f['complete'])
            np.testing.assert_array_equal(f['qualified'], (f['status'] == 1)&np.all(f['final_returns'] == 1, axis=1))
            values[ids], q[ids], seen[ids] = f['values'], f['qualified'], True
    assert seen.all(); values, q = values.reshape(n, 4, 5, 27), q.reshape(n, 4, 5)
    catalogue = json.loads((ISO/'x8_sfr_validation_20261102/catalogue.json').read_text())['bins']
    checked = 0; point_error = 0.; se_error = 0.
    profiles = {(x['cohort'], x['mode'], x['branch'], x['region']): x for x in review['profiles']}
    effects = {(x['cohort'], x['mode'], x['effect'], x.get('branch'), x['region']): x for x in review['paired_effects']}
    shapes = {(x['cohort'], x['mode'], x['branch'], x['contrast']): x for x in review['shape_checks']}
    slopes = {(x['mode'], x['branch'], x['window_Myr'], str(x['radial'])): x for x in review['sfr_slopes']}
    def coefficients(window, radial):
        a = np.zeros(len(catalogue)); denominator = 0.
        for r in range(1, 8) if radial == 'fixed_effect' else [0]:
            ids = [i for i, x in enumerate(catalogue) if x['eligible'] and x['window_Myr'] == window and x['radial_bin'] == r]
            if len(ids) < (2 if radial == 'fixed_effect' else 3): continue
            x = np.array([(catalogue[i]['log_sfr_lo']+catalogue[i]['log_sfr_hi'])/2 for i in ids]); dx = x-x.mean()
            a[ids] = dx; denominator += np.sum(dx*dx)
        assert denominator > 0
        return a/denominator
    with h5py.File(SHIELD/'sample.h5') as f:
        for cohort in ('global', 'outer'):
            g = f[cohort]; start = int(g.attrs['offset']); size = len(g['mass']); sl = slice(start, start+size)
            strata, population, pi = g['stratum'][:], g['population'][:], g['pi'][:]
            allocation = np.bincount(strata, minlength=len(population))
            np.testing.assert_array_equal(pi, allocation[strata]/population[strata])
            groups = [(np.flatnonzero(strata == h), population[h]) for h in np.unique(strata)]
            weight = g['nuclei'][:]/pi; kernels = g['kernel'][:]; cache = {}
            def check(record, point, score):
                nonlocal checked, point_error, se_error
                if record['point'] is None:
                    assert not np.isfinite(point); checked += 1; return
                variance = 0.
                for ids, N in groups:
                    nh = len(ids)
                    if nh < N:
                        centred = score[ids]-np.mean(score[ids])
                        variance += nh*(1-nh/N)*np.dot(centred, centred)/(nh-1)
                se = np.sqrt(max(0., variance))
                point_error = max(point_error, abs(point-record['point']))
                se_error = max(se_error, abs(se-record['standard_error']))
                np.testing.assert_allclose([point, se], [record['point'], record['standard_error']], rtol=0, atol=4e-12)
                np.testing.assert_allclose(record['interval_95'], [point-1.96*se, point+1.96*se], rtol=0, atol=1e-11)
                checked += 1
            def measure(mode, branch, mask, op, sfr=False):
                key = (mode, branch, mask.tobytes(), op, sfr)
                if key not in cache:
                    k = g['sfr_kernel'][:] if sfr else kernels[:, op]
                    d = np.where(mask[:, None], values[sl, mode, branch, 7:9], 0.)
                    wk = weight[:, None]*k; sums = np.sum(wk[:, :, None]*d[:, None, :], axis=0)
                    good = np.all(sums > 0, axis=1)
                    point = np.full(len(sums), np.nan); score = np.zeros_like(wk)
                    point[good] = np.log10(sums[good, 0])-np.log10(sums[good, 1])
                    score[:, good] = wk[:, good]*(d[:, 0, None]/sums[None, good, 0]-d[:, 1, None]/sums[None, good, 1])/np.log(10)
                    cache[key] = point, score
                return cache[key]
            for m, mode in enumerate(source.MODES):
                for b, branch in enumerate(source.BRANCHES):
                    op = int(b >= 2); mask = q[sl, m, b]
                    x, sx = measure(m, b, mask, op)
                    for j, region in enumerate(source.REGIONS): check(profiles[cohort, mode, branch, region]['log_energy_ratio'], x[j], sx[:, j])
                    if m:
                        mask = q[sl, m, b]&q[sl, 0, b]
                        x, sx = measure(m, b, mask, op); y, sy = measure(0, b, mask, op)
                        for j, region in enumerate(source.REGIONS): check(effects[cohort, mode, 'absolute_ratio_shift', branch, region]['change_dex'], x[j]-y[j], sx[:, j]-sy[:, j])
                for effect, (a, b) in source.PAIRS.items():
                    op = int(effect != 'source'); mask = q[sl, m, a]&q[sl, m, b]&q[sl, 0, a]&q[sl, 0, b]
                    x, sx = measure(m, a, mask, op); y, sy = measure(m, b, mask, op)
                    zx, szx = measure(0, a, mask, op); zy, szy = measure(0, b, mask, op)
                    for j, region in enumerate(source.REGIONS):
                        row = effects[cohort, mode, effect, None, region]
                        check(row['paired_response_dex'], x[j]-y[j], sx[:, j]-sy[:, j])
                        check(row['response_change_dex'], x[j]-y[j]-zx[j]+zy[j], sx[:, j]-sy[:, j]-szx[:, j]+szy[:, j])
                for b in (0, 4):
                    mask = q[sl, m, b]&q[sl, 0, b]
                    x, sx = measure(m, b, mask, int(b >= 2)); y, sy = measure(0, b, mask, int(b >= 2))
                    for key, row in shapes.items():
                        if key[:3] != (cohort, mode, source.BRANCHES[b]): continue
                        a = np.zeros(19); name = key[3]
                        if name.startswith('annulus_'):
                            j = int(name.split('_')[1])-1; a[j:j+2] = [-1, 1]
                        elif name.startswith('outer_endpoint'): a[8], a[14] = -1, 1
                        else:
                            dx = np.arange(8.5, 15)-11.5; a[8:15] = dx/np.sum(dx*dx)
                        use = a != 0
                        check(row['direct'], a[use]@x[use], sx@a)
                        check(row['change'], a[use]@(x-y)[use], (sx-sy)@a)
                if cohort == 'global':
                    mask = q[sl, m][:, [0, 2, 3, 4]].all(axis=1)&q[sl, 0][:, [0, 2, 3, 4]].all(axis=1)
                    for b in (0, 2, 3, 4):
                        x, sx = measure(m, b, mask, 1, True); y, sy = measure(0, b, mask, 1, True)
                        for window in (1, 10):
                            for radial in (0, 'fixed_effect'):
                                a = coefficients(window, radial); use = a != 0
                                row = slopes[mode, source.BRANCHES[b], window, str(radial)]
                                check(row['direct'], a[use]@x[use], sx@a)
                                check(row['change'], a[use]@(x-y)[use], (sx-sy)@a)
            print('Independently audited shielding cohort', cohort, flush=True)
    whole = [x for x in review['paired_effects'] if x['cohort'] == 'global' and x['region'] == 'full_field' and x['effect'] in ('source', 'exchange', 'uniform_exchange')]
    outer = [x for x in review['shape_checks'] if x['cohort'] == 'outer' and x['contrast'] == 'outer_OLS_slope_dex_per_kpc']
    sfr = [x for x in review['sfr_slopes'] if x['radial'] == 'fixed_effect']
    assert all(x['paired_response_dex']['interval_95'][1] < 0 if x['effect'] == 'source' else x['paired_response_dex']['interval_95'][0] > 0 for x in whole)
    assert all(x['direct']['interval_95'][0] > 0 for x in outer)
    assert all(x['direct']['interval_95'][1] < 0 for x in sfr)
    source.save(ROOT/'shielding_audit.json', dict(passed=True, checked_interval_sets=checked,
        maximum_point_difference=point_error, maximum_standard_error_difference=se_error,
        input_review_sha256=sha(SHIELD/'review.json'), qualified_states=review['qualified_states'], logical_states=review['logical_states'],
        global_response_signs_supported=True, outer_positive_slope_intervals=len(outer), negative_sfr_intervals=len(sfr),
        maximum_absolute_global_response_shift_dex=max(abs(x['response_change_dex']['point']) for x in whole),
        selected_global_responses=whole, outer_slopes=outer,
        scope='Conditional solved-state carbon-lookup stress tests only; fixed oxygen 557 and off-table dependence remain untested.'))
    print('Independent shielding audit passed:', checked, 'interval sets.', flush=True)


if __name__ == '__main__': main()
