"""Reconstruct selection science from direct residual references, without fitting."""
import json
import os
import sys

import h5py
import numpy as np

import closure_common as c
sys.path.insert(0, str(c.ISO/'x8_selection_reference_20261108'))
from selection_design import selected_bounds
from selection_statistics import detection, summaries
from selection_bootstrap import replicate_weights


def quantile_bounds(bounds):
    x = np.asarray(bounds)
    lo = np.quantile(np.where(np.isfinite(x[..., 0]), x[..., 0], -np.inf), .0125, axis=0, method='lower')
    hi = np.quantile(np.where(np.isfinite(x[..., 1]), x[..., 1], np.inf), .9875, axis=0, method='higher')
    return np.stack((lo, hi), axis=-1)


def spread_bounds(bounds):
    x = np.log10(bounds)
    if not np.isfinite(x).all():return np.array([np.nan, np.nan])
    return np.array([max(0., x[:, 0].max()-x[:, 1].min()), x[:, 1].max()-x[:, 0].min()])


def main():
    assert int(os.environ.get('PBS_NCPUS', '0')) > 0
    manifest = c.verify(); source = c.ISO/'x8_selection_reference_20261108'
    audit = json.loads((source/'independent_audit.json').read_text())
    assert audit['passed'] and audit['review_sha256'] == c.sha(source/'review.json')
    with np.load(source/'aperture_results.npz') as f:
        total, low, high, xy, weight = (f[k] for k in ('estimates', 'reference_low', 'reference_high', 'xy', 'pixel_weight'))
    with h5py.File(source/'sample.h5') as f:
        strata = f['apertures/stratum'][:]; population = f['apertures/population'][:]
    radius = np.hypot(*xy.T); rng = np.random.default_rng(c.SEED+1)
    rows = []; contrasts = []; depths = [100., 1000., 10000., 100000., 1000000.]
    for radial in range(8, 15):
        use = (radius >= radial) & (radius < radial+1)
        w = weight[use]; boot = replicate_weights(strata[use], population, rng, 800)*w[None, :]
        for model, name in enumerate(c.MODELS):
            line, lo, hi = total[use, model, :2], low[use, model, :2], high[use, model, :2]
            keep = []; point_by_depth = []; replicas_by_depth = []
            for depth in depths:
                certain = detection(lo, depth); possible = detection(hi, depth)
                eligible = int(certain.sum()) >= 10
                point = summaries(line, w, detection(line, depth))
                bounds = selected_bounds(lo, hi, w, depth, 'dual')
                replicas = np.array([selected_bounds(lo, hi, bw, depth, 'dual') for bw in boot]) if eligible else np.full((800, 4, 2), np.nan)
                interval = quantile_bounds(replicas) if eligible else np.full((4, 2), np.nan)
                rows.append(dict(model=name, annulus_kpc=[radial, radial+1], depth=depth,
                    observables=['median_local_ratio', 'ratio_of_line_medians', 'ratio_of_line_sums', 'detected_fraction'],
                    direct_point=point, reference_bounds=bounds, approximate_interval_95=interval,
                    certain_detections=int(certain.sum()), possible_detections=int(possible.sum()), eligible=eligible))
                if eligible and np.all(bounds[:3] > 0) and np.isfinite(bounds[:3]).all():
                    keep.append(depth); point_by_depth.append(point); replicas_by_depth.append(replicas)
            if len(keep) < 2:continue
            point_by_depth = np.asarray(point_by_depth); replicas_by_depth = np.asarray(replicas_by_depth)
            # Identical bootstrap weights at every depth retain cross-depth
            # dependence. Bound maxima/minima before taking quantiles.
            spreads = np.array([[spread_bounds(replicas_by_depth[:, b, j]) for j in range(3)] for b in range(800)])
            spread_ci = quantile_bounds(spreads)
            median_minus_sums = np.column_stack((spreads[:, 0, 0]-spreads[:, 2, 1], spreads[:, 0, 1]-spreads[:, 2, 0]))
            medians_minus_sums = np.column_stack((spreads[:, 1, 0]-spreads[:, 2, 1], spreads[:, 1, 1]-spreads[:, 2, 0]))
            contrasts.append(dict(model=name, annulus_kpc=[radial, radial+1], compared_depths=keep,
                direct_depth_spread_dex=np.ptp(np.log10(point_by_depth[:, :3]), axis=0),
                spread_approximate_intervals_95=spread_ci,
                median_spread_minus_sum_spread_interval_95=quantile_bounds(median_minus_sums),
                ratio_of_medians_spread_minus_sum_spread_interval_95=quantile_bounds(medians_minus_sums)))
    c.save(c.ROOT/'selection_direct_review.json', dict(complete=True, manifest_sha256=manifest,
        source_review_sha256=c.sha(source/'review.json'), rows=rows, depth_contrasts=contrasts,
        fitting=False, scientific_acceptance=False,
        scope='Design-weighted direct-reference curves; joint depth resampling; matched qualified population. Approximate pointwise intervals preserve reference bands and selection uncertainty. Failed-state light and physical error excluded.'))
    print('Direct selection curves and depth contrasts reconstructed.', flush=True)


if __name__ == '__main__':main()
