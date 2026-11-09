"""Compare direct and frozen-surrogate aperture selection and quantiles."""
import json
import os

import h5py
import numpy as np
from scipy.sparse import coo_matrix

import aperture_common as c
from aperture_statistics import detection, summaries


def main():
    assert int(os.environ.get('PBS_NCPUS', '0')) > 0
    manifest, ready = c.ready(); assert not (c.ROOT/'review.json').exists()
    with h5py.File(c.ROOT/'census.h5') as f:
        n = len(f['particles/mass']); xy = f['apertures/xy'][:]; radius = np.hypot(*xy.T)
        strata = f['apertures/stratum'][:]; weight = 1/f['apertures/inclusion_probability'][:]
        populations, allocations = f['apertures/population'][:], f['apertures/allocation'][:]
        np.testing.assert_array_equal(np.bincount(strata, minlength=len(populations)), allocations)
        particle, aperture, kernel = [f['operator/'+k][:] for k in ('particle', 'aperture', 'weight')]
        mass, nuclei = f['particles/mass'][:], f['particles/nuclei'][:]
        prediction, factors = f['prediction'][:], f['conversion_line'][:]
        full_line = f['apertures/cached_line'][:]
    values = np.full((n*5, 27), np.nan); qualified = np.zeros(n*5, bool); seen = qualified.copy()
    for part in range(c.PARTITIONS):
        with np.load(c.ROOT/f'direct/part{part:02d}.npz') as f:
            assert bool(f['complete']) and json.loads(str(f['metadata']))['census_sha256'] == ready['census_sha256']
            ids = f['task_ids']; assert not seen[ids].any()
            np.testing.assert_array_equal(f['qualified'], (f['status'] == 1)&np.all(f['final_returns'] == 1, axis=1))
            values[ids], qualified[ids], seen[ids] = f['values'], f['qualified'], True
    assert seen.all()
    values, qualified = values.reshape(n, 5, 27), qualified.reshape(n, 5)
    assert np.isfinite(values[qualified]).all()
    direct = np.zeros_like(full_line); proxy = np.zeros_like(full_line)
    dm = np.zeros_like(full_line); pm = np.zeros_like(full_line); missing = np.zeros_like(full_line)
    matrices = [coo_matrix((kernel[:, op], (aperture, particle)), shape=(len(xy), n)).tocsr() for op in range(2)]
    independent_error = 0.
    for model in range(6):
        b = c.MODEL_BRANCH[model]
        q = qualified[:, b] if model < 2 else np.all(qualified[:, [0, 2, 3, 4]], axis=1)
        matrix = matrices[int(model >= 2)]
        direct[:, model] = (matrix@(nuclei[:, None]*np.where(q[:, None], values[:, b, 7:9], 0.)))*factors
        proxy[:, model] = (matrix@(nuclei[:, None]*np.where(q[:, None], prediction[:, model, 4:6], 0.)))*factors
        dm[:, model] = matrix@(mass[:, None]*np.where(q[:, None], values[:, b, 4:6], 0.))
        pm[:, model] = matrix@(mass[:, None]*np.where(q[:, None], prediction[:, model, 1:3], 0.))
        all_proxy = (matrix@(nuclei[:, None]*prediction[:, model, 4:6]))*factors
        np.testing.assert_allclose(all_proxy, full_line[:, model], rtol=3e-7, atol=1e-20)
        missing[:, model] = np.divide(all_proxy-proxy[:, model], all_proxy,
            out=np.zeros_like(all_proxy), where=all_proxy > 0)
        # A separate accumulation checks sparse orientation, q masks,
        # duplicate aperture contributions and the physical unit conversion.
        qvalues = np.where(q[:, None], values[:, b, 7:9], 0.)
        for line in range(2):
            independent = np.bincount(aperture, weights=kernel[:, int(model >= 2)]*
                nuclei[particle]*qvalues[particle, line], minlength=len(xy))*factors[line]
            np.testing.assert_allclose(independent, direct[:, model, line], rtol=2e-12, atol=1e-22)
            independent_error = max(independent_error, float(np.max(np.abs(independent-direct[:, model, line]))))
    c.base.atomic_npz(c.ROOT/'aperture_results.npz', xy=xy, stratum=strata, pixel_weight=weight,
        direct_line=direct, matched_surrogate_line=proxy, population_surrogate_line=full_line,
        direct_molecule=dm, matched_surrogate_molecule=pm, missing_predicted_line_fraction=missing)
    rows = []; migration = []; pixel_errors = []; rng = np.random.default_rng(c.SEED+1)
    stats = ('median_local_ratio', 'ratio_of_line_medians', 'ratio_of_line_sums', 'detection_fraction')
    for radial in range(1, 15):
        use = np.flatnonzero((radius >= radial)&(radius < radial+1))
        assert len(use) == (8 if radial < 8 else 32)
        w = weight[use]
        # Resample within the original brightness strata. Shared aperture
        # draws retain covariance between direct and surrogate statistics.
        boot = []
        for rep in range(1000):
            boot.append(np.concatenate([rng.choice(np.flatnonzero(strata[use] == h), np.sum(strata[use] == h), replace=True)
                for h in np.unique(strata[use])]))
        for model, name in enumerate(c.MODELS):
            dl, pl = direct[use, model], proxy[use, model]
            valid = np.all(dl > 0, axis=1)&np.all(pl > 0, axis=1)
            err = np.full((len(use), 3), np.nan)
            err[valid, :2] = np.log10(dl[valid]/pl[valid])
            err[valid, 2] = err[valid, 0]-err[valid, 1]
            pixel_errors.append(dict(model=name, annulus_kpc=[radial, radial+1], apertures=len(use),
                valid_apertures=int(valid.sum()), log_line_and_ratio_errors=err,
                maximum_missing_predicted_line_fraction=np.max(missing[use, model], axis=0)))
            for mode in ('dual', 'dual_and_ratio_snr'):
                for depth in c.DEPTHS:
                    md, mp = detection(dl, depth, mode), detection(pl, depth, mode)
                    changed = md != mp
                    migration.append(dict(model=name, annulus_kpc=[radial, radial+1], selection=mode, depth=depth,
                        direct_detections=int(md.sum()), proxy_detections=int(mp.sum()),
                        weighted_changed_fraction=float(w[changed].sum()/w.sum()),
                        weighted_false_positive_fraction=float(w[(~md)&mp].sum()/w.sum()),
                        weighted_false_negative_fraction=float(w[md&(~mp)].sum()/w.sum()),
                        scope='Matched successful-state maps; threshold applied separately.'))
                    ds, ps = summaries(dl, w, md), summaries(pl, w, mp)
                    # Ten independently drawn positions is only a reporting
                    # floor, never a precision or scientific acceptance gate.
                    eligible = md.sum() >= 10 and mp.sum() >= 10
                    delta = np.r_[np.log10(ds[:3]/ps[:3]), ds[3]-ps[3]] if eligible else np.full(4, np.nan)
                    replicas = []
                    if eligible:
                        for ix in boot:
                            one, two = summaries(dl[ix], w[ix], md[ix]), summaries(pl[ix], w[ix], mp[ix])
                            replicas.append(np.r_[np.log10(one[:3]/two[:3]), one[3]-two[3]])
                    intervals = np.nanquantile(replicas, [.025, .975], axis=0).T if eligible else np.full((4, 2), np.nan)
                    for j, statistic in enumerate(stats):
                        ci = intervals[j]
                        rows.append(dict(model=name, annulus_kpc=[radial, radial+1], selection=mode, depth=depth,
                            statistic=statistic, direct=ds[j], matched_surrogate=ps[j], direct_minus_surrogate=delta[j],
                            error_unit='fraction' if j == 3 else 'dex', bootstrap_interval_95=ci,
                            sufficient_sample_for_reporting=bool(eligible),
                            interval_within_0p05_dex=bool(eligible and j < 3 and np.all(np.isfinite(ci)) and ci[0] >= -.05 and ci[1] <= .05)))
    # Common control detection is also formed separately on direct and proxy
    # maps, so it does not silently freeze the surrogate-selected catalogue.
    common = []
    for radial in range(1, 15):
        use = (radius >= radial)&(radius < radial+1); w = weight[use]
        md = np.all(np.stack([detection(direct[use, m], 1000) for m in range(2, 6)]), axis=0)
        mp = np.all(np.stack([detection(proxy[use, m], 1000) for m in range(2, 6)]), axis=0)
        for model in range(2, 6):
            def ratios(line, molecule, mask, weights):
                ls = (weights[:, None]*mask[:, None]*line).sum(0)
                ms = (weights[:, None]*mask[:, None]*molecule).sum(0)
                if np.any(ls <= 0) or np.any(ms <= 0):
                    return [np.nan]*3
                l, m = ls[0]/ls[1], ms[0]/ms[1]
                return [l, m, l/m]
            one = np.asarray(ratios(direct[use, model], dm[use, model], md, w))
            two = np.asarray(ratios(proxy[use, model], pm[use, model], mp, w))
            eligible = md.sum() >= 10 and mp.sum() >= 10
            intervals = np.full((3, 2), np.nan)
            if eligible:
                replicates = []
                s = strata[use]
                for rep in range(1000):
                    ix = np.concatenate([rng.choice(np.flatnonzero(s == h), np.sum(s == h), replace=True) for h in np.unique(s)])
                    d = ratios(direct[use, model][ix], dm[use, model][ix], md[ix], w[ix])
                    p = ratios(proxy[use, model][ix], pm[use, model][ix], mp[ix], w[ix])
                    replicates.append(np.log10(np.asarray(d)/np.asarray(p)))
                intervals = np.nanquantile(replicates, [.025, .975], axis=0).T
            common.append(dict(model=c.MODELS[model], annulus_kpc=[radial, radial+1],
                direct_line_molecule_effective=one, proxy_line_molecule_effective=two,
                direct_minus_proxy_dex=np.log10(one/two), paired_bootstrap_interval_95=intervals,
                sufficient_sample_for_reporting=bool(eligible),
                direct_detections=int(md.sum()), proxy_detections=int(mp.sum()),
                weighted_changed_fraction=float(w[md != mp].sum()/w.sum())))
    c.save(c.ROOT/'review.json', dict(complete=True, manifest_sha256=manifest, census_sha256=ready['census_sha256'],
        sampled_apertures=len(xy), unique_particles=n, qualified_by_branch=qualified.sum(0),
        pixel_errors=pixel_errors, selection_migration=migration, selected_statistics=rows, common_D1000=common,
        physical_model_uncertainty_included=False, full_population_accuracy_claim=False,
        model_training_performed=False, bootstrap_replicates=1000,
        scope='Particle census within probability-sampled apertures; production branches use their own strict success mask, '
            'while four controls share a common particle-success mask. '
            'Stratified paired pixel bootstrap approximates aperture-sampling uncertainty; unknown failed-state light is not bounded. '
            'No automatic manuscript acceptance; inner apertures are guards and not quantile validation.'))
    c.save(c.ROOT/'independent_audit.json', dict(passed=True, independently_checked_line_totals=12*len(xy),
        maximum_absolute_difference=independent_error, review_sha256=c.sha(c.ROOT/'review.json'),
        scope='Independent particle accumulation; not a physical or missing-state luminosity bound.'))
    print('Aperture validation reviewed;', len(rows), 'statistic records.', flush=True)


if __name__ == '__main__':
    main()
