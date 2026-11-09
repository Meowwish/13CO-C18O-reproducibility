"""Frozen SFR audit lineage and reusable, testable statistical functions."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
ISO = ROOT.parent
SOURCE = ISO/'x8_bounded_reference_20261101'
PARENT = ISO/'x8_failure_diagnostics_20261102'
DEFAULT = ISO/'x8_reference_recovery_20261101'
sys.path.insert(0, str(SOURCE))
import common as base
sys.path.insert(0, str(PARENT))
import diagnostics_common as diagnostic
sys.path.insert(0, str(ISO/'highres_snapshot575_temporal'))
sys.path.insert(0, str(ISO))

MAPS = dict(Full=ISO/'highres_snapshot575_temporal/output/snapshot_575_maps_125pc.npz',
            SLUG=ISO/'highres_snapshot575_temporal/slug_only/output/snapshot_575_maps_125pc.npz')
SFR = ISO/'highres_snapshot575_temporal/output/sfr/snapshot575_SFR_maps.npz'
PAPER_AUDIT = ISO/'paper_completion_20261024/sfr_checks.json'
WINDOWS = (1, 10)
EDGES = np.arange(-3., -.49, .5)
LABELS = ('default_reference', 'resource_extended')


def save(path, value):
    base.save(path, value)


def verify():
    manifest = json.loads((ROOT/'manifest.json').read_text())
    for path, sha in manifest['hashes'].items():
        assert base.core.digest(path) == sha, path
    assert diagnostic.verify() == manifest['parent_manifest_sha256']
    assert base.core.digest(SOURCE/'sample.h5') == manifest['source_sample_sha256']
    return base.core.digest(ROOT/'manifest.json')


def ready():
    sha = verify()
    record = json.loads((ROOT/'prepared.json').read_text())
    assert record['complete'] and record['manifest_sha256'] == sha
    for name, key in (('adjoint_weights.npy', 'weights_sha256'), ('catalogue.json', 'catalogue_sha256')):
        assert base.core.digest(ROOT/name) == record[key]
    return sha, record


def groups(strata, population):
    return [(np.flatnonzero(strata == h), int(population[h])) for h in np.unique(strata)]


def variance(scores, design):
    scores = np.asarray(scores)
    total = np.zeros(scores.shape[1:])
    for indices, N in design:
        nh = len(indices)
        assert 0 < nh <= N
        if nh < N:
            assert nh >= 2
            total += nh*(1-nh/N)*np.var(scores[indices], axis=0, ddof=1)
    return total


def interval(point, score, design):
    if not np.isfinite(point) or not np.all(np.isfinite(score)):
        return dict(point=None, standard_error=None, interval_95=None)
    se = float(np.sqrt(max(0., float(variance(score, design)))))
    return dict(point=float(point), standard_error=se, interval_95=[float(point-1.96*se), float(point+1.96*se)])


def coefficients(catalogue, window, radial):
    """Fixed OLS or within-annulus coefficients, independent of direct labels."""
    coefficient = np.zeros(len(catalogue))
    selected = [i for i, row in enumerate(catalogue) if row['window_Myr'] == window and row['eligible']]
    def assign(indices):
        x = np.array([(catalogue[i]['log_sfr_lo']+catalogue[i]['log_sfr_hi'])/2 for i in indices])
        dx = x-x.mean()
        coefficient[indices] = dx
        return float(dx@dx)
    if radial == 'fixed_effect':
        denominator = 0.
        count = 0
        for r in range(1, 8):
            use = [i for i in selected if catalogue[i]['radial_bin'] == r]
            if len(use) >= 2:
                denominator += assign(use)
                count += 1
        if count < 2 or denominator <= 0:
            return None
    else:
        use = [i for i in selected if catalogue[i]['radial_bin'] == radial]
        if len(use) < 3:
            return None
        denominator = assign(use)
        if denominator <= 0:
            return None
    coefficient /= denominator
    assert abs(coefficient.sum()) < 1e-12
    return coefficient


def slope_and_score(coefficient, totals, line_values, kernel, base_weight):
    """Derivative keeps cross-bin covariance from each shared particle."""
    use = coefficient != 0
    if not use.any() or np.any(totals[use] <= 0) or not np.all(np.isfinite(totals[use])):
        return None, None
    a = coefficient[use]
    t = totals[use]
    value = float(a @ np.log10(t[:, 0]/t[:, 1]))
    score = base_weight*(line_values[:, 0]*(kernel[:, use]@(a/t[:, 0]))
                        -line_values[:, 1]*(kernel[:, use]@(a/t[:, 1])))/np.log(10)
    return value, score
