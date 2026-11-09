"""Design-weighted radial errors, paired responses and correlated shapes."""
import json
import os

import h5py
import numpy as np
from scipy.stats import norm

import outer_common as c

PAIRS = dict(source=(1, 0), exchange=(2, 3), uniform_exchange=(4, 5),
             isotope_pattern_on=(2, 4), isotope_pattern_off=(3, 5))


def ratio_measure(direct, proxy, qualified, kernels, weight, strata, population, allocation):
    """Returns joint radial scores; all sampled strata remain in variance."""
    q = np.asarray(qualified, bool)
    d = np.where(q[:, None], direct, 0.)
    wk = kernels*weight[:, None]
    D = wk.T@d
    P = (wk*q[:, None]).T@proxy
    all_proxy = wk.T@proxy
    missing = (wk*(~q)[:, None]).T@np.maximum(proxy, 0.)
    positive_all = wk.T@np.maximum(proxy, 0.)
    good = np.all(D > 0, axis=1) & np.all(P > 0, axis=1)
    rd = np.full(len(D), np.nan)
    rp = np.full(len(D), np.nan)
    sd = np.zeros_like(kernels)
    sp = np.zeros_like(kernels)
    rd[good] = np.log10(D[good, 0]/D[good, 1])
    rp[good] = np.log10(P[good, 0]/P[good, 1])
    sd[:, good] = wk[:, good]*(d[:, 0, None]/D[None, good, 0]-d[:, 1, None]/D[None, good, 1])/np.log(10)
    sp[:, good] = wk[:, good]*q[:, None]*(proxy[:, 0, None]/P[None, good, 0]-proxy[:, 1, None]/P[None, good, 1])/np.log(10)
    cd = c.covariance(sd, strata, population, allocation)
    ce = c.covariance(sd-sp, strata, population, allocation)
    rows = []
    for j, name in enumerate(c.REGIONS):
        light = wk[:, j, None]*np.maximum(d, 0.)
        sums = light.sum(0)
        square = (light**2).sum(0)
        ess = np.divide(sums**2, square, out=np.zeros(2), where=square > 0)
        largest = np.divide(light.max(0), sums, out=np.zeros(2), where=sums > 0)
        miss = np.divide(missing[j], positive_all[j], out=np.full(2, np.nan), where=positive_all[j] > 0)
        err = c.interval(rd[j]-rp[j], ce[j, j])
        ci = err['interval_95']
        rows.append(dict(region=name, direct_log_energy_ratio=c.interval(rd[j], cd[j, j]),
            direct_minus_surrogate_dex=err, matched_surrogate_log_energy_ratio=rp[j],
            missing_predicted_positive_line_fractions=miss, effective_line_samples=ess,
            largest_positive_line_weight_fraction=largest,
            contributing_reference_particles=int(np.sum(q & (kernels[:, j] > 0))),
            interval_within_0p05_dex=bool(ci is not None and ci[0] >= -.05 and ci[1] <= .05)))
    return dict(rows=rows, direct=rd, proxy=rp, direct_score=sd, proxy_score=sp,
                direct_covariance=cd, error_covariance=ce)


def main():
    assert int(os.environ.get('PBS_NCPUS', '0')) > 0
    sha, ready = c.ready()
    assert not (c.ROOT/'review.json').exists()
    with h5py.File(c.ROOT/'sample.h5') as f:
        n = len(f['sample/mass'])
        weight = f['sample/nuclei'][:]/f['sample/inclusion_probability'][:]
        strata = f['sample/stratum'][:]
        population, allocation = f['stratum_population'][:], f['stratum_sample'][:]
        proxy = f['sample/lum_per_H'][:, :, 1:3]
    values = np.full((n*5, 27), np.nan)
    qualified = np.zeros(n*5, bool)
    seen = np.zeros(n*5, bool)
    for part in range(c.PARTITIONS):
        with np.load(c.ROOT/f'direct/part{part:02d}.npz') as f:
            meta = json.loads(str(f['metadata']))
            assert meta['manifest_sha256'] == sha and meta['sample_sha256'] == ready['sample_sha256'] and bool(f['complete'])
            ids = f['task_ids']
            assert not seen[ids].any() and np.all(ids % c.PARTITIONS == part)
            values[ids], qualified[ids], seen[ids] = f['values'], f['qualified'], True
            np.testing.assert_array_equal(f['qualified'], (f['status'] == 1) & np.all(f['final_returns'] == 1, axis=1))
    assert seen.all() and np.all(np.isfinite(values[qualified]))
    values, qualified = values.reshape(n, 5, 27), qualified.reshape(n, 5)
    kernel = np.full((n, 4, len(c.REGIONS)), np.nan)
    seen = np.zeros(n, bool)
    for part in range(4):
        with np.load(c.ROOT/f'geometry/part{part:02d}.npz') as f:
            assert str(f['manifest_sha256']) == sha and bool(f['complete'])
            assert str(f['sample_sha256']) == ready['sample_sha256']
            ids = f['ids']
            assert not seen[ids].any() and np.all(ids % 4 == part)
            kernel[ids], seen[ids] = f['kernel'], True
    assert seen.all() and np.all(np.isfinite(kernel)) and np.all(kernel >= 0)
    gr = json.loads((c.ROOT/'geometry_ready.json').read_text())
    assert c.digest(c.ROOT/'exact_maps.npz') == gr['exact_maps_sha256']
    with np.load(c.ROOT/'exact_maps.npz') as f:
        exact = f['totals']
    config = json.loads((c.ISO/'highres_snapshot575_temporal/config.json').read_text())
    freq = config['projection']['line_frequencies_Hz']
    offset = 3*np.log10(freq['C18O']/freq['13CO'])
    result = dict(complete=True, manifest_sha256=sha, sample_sha256=ready['sample_sha256'],
        independent_sample_particles=n, qualified_states_by_branch=qualified.sum(0),
        model_training_performed=False, full_population_accuracy_claim=False,
        scope='Independent stratified SRSWOR, matched successful direct states; unknown failed-state light is not bounded.',
        ratio_offset_energy_to_brightness_dex=offset, radial_targets=c.RADIAL_TARGETS,
        model_checks=[], paired_checks=[], shape_checks=[], covariance={}, scientific_acceptance=False)
    for geometry in range(2):
        label = ('native', 'lattice')[geometry]
        def measure(model, q):
            op = geometry*2+int(model >= 2)
            return ratio_measure(values[:, c.MODEL_BRANCH[model], 7:9], proxy[:, model], q,
                                 kernel[:, op], weight, strata, population, allocation)
        for model, name in enumerate(c.MODELS):
            q = qualified[:, c.MODEL_BRANCH[model]]
            m = measure(model, q)
            for j, row in enumerate(m['rows']):
                exact_ratio = exact[geometry, j, model, 0]/exact[geometry, j, model, 1] if exact[geometry, j, model, 1] > 0 else None
                result['model_checks'].append(dict(geometry=label, model=name,
                    exact_population_surrogate_brightness_ratio=exact_ratio, **row))
            result['covariance'][label+'/'+name] = dict(direct=m['direct_covariance'], error=m['error_covariance'])
            # Every adjacent contrast is reported; familywise intervals prevent
            # choosing a visually prominent dip or rise after inspecting labels.
            contrasts = []
            for r in range(14):
                a = np.zeros(len(c.REGIONS)); a[r:r+2] = [-1, 1]
                contrasts.append((f'annulus_{r+1}_{r+2}_minus_{r}_{r+1}', a, 14))
            a = np.zeros(len(c.REGIONS)); a[8], a[14] = -1, 1
            contrasts.append(('outer_endpoint_14_15_minus_8_9', a, 1))
            a = np.zeros(len(c.REGIONS)); x = np.arange(8.5, 15.)
            a[8:15] = (x-x.mean())/np.sum((x-x.mean())**2)
            contrasts.append(('outer_OLS_slope_dex_per_kpc', a, 1))
            for title, a, family in contrasts:
                use = a != 0
                point = float(a[use]@m['direct'][use])
                pred = float(a[use]@m['proxy'][use])
                variance = float(a@m['direct_covariance']@a)
                error_variance = float(a@m['error_covariance']@a)
                result['shape_checks'].append(dict(geometry=label, model=name, contrast=title,
                    direct=c.interval(point, variance), matched_surrogate=pred,
                    direct_minus_surrogate=c.interval(point-pred, error_variance),
                    within_curve_familywise_95=c.interval(point, variance, float(norm.isf(.05/(2*family)))),
                    physical_model_uncertainty_included=False))
        for name, (a, b) in PAIRS.items():
            q = qualified[:, c.MODEL_BRANCH[a]] & qualified[:, c.MODEL_BRANCH[b]]
            ma, mb = measure(a, q), measure(b, q)
            d = ma['direct']-mb['direct']; p = ma['proxy']-mb['proxy']
            sd = ma['direct_score']-mb['direct_score']
            se = sd-ma['proxy_score']+mb['proxy_score']
            vd = np.diag(c.covariance(sd, strata, population, allocation))
            ve = np.diag(c.covariance(se, strata, population, allocation))
            for j, region in enumerate(c.REGIONS):
                result['paired_checks'].append(dict(geometry=label, comparison=name, region=region,
                    direct_response_dex=c.interval(d[j], vd[j]), matched_surrogate_response_dex=p[j],
                    response_error_dex=c.interval(d[j]-p[j], ve[j]),
                    missing_predicted_positive_line_fractions=[ma['rows'][j]['missing_predicted_positive_line_fractions'],
                                                              mb['rows'][j]['missing_predicted_positive_line_fractions']]))
        print('Reviewed', label, flush=True)
    c.save(c.ROOT/'review.json', result)
    rows = ['# Outer-disc direct-reference validation', '',
            'Conditional on successful states; intervals quantify sampling uncertainty only.', '',
            '| Geometry | Model | Annulus kpc | direct - surrogate dex | 95% interval | ESS 13/18 |',
            '|---|---|---|---:|---|---|']
    for r in result['model_checks']:
        if r['region'] not in c.REGIONS[8:15]:
            continue
        error = r['direct_minus_surrogate_dex']
        rows.append(f"| {r['geometry']} | {r['model']} | {r['region']} | {error['point']} | {error['interval_95']} | {r['effective_line_samples']} |")
    rows += ['', 'See review.json for correlated shapes, paired responses, unresolved-light diagnostics and all radii.',
             'No manuscript acceptance or automatic training is performed.']
    (c.ROOT/'RESULTS.md').write_text('\n'.join(rows)+'\n')


if __name__ == '__main__':
    main()
