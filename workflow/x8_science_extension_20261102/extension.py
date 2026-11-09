"""Experiment-local helpers; importing this module performs no chemistry."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
ISO = ROOT.parent
SOURCE = ISO/'x8_bounded_reference_20261101'
for folder in ('x8_full_direct_575_20261028', 'x8_bounded_reference_20261101',
               'x8_reference_recovery_20261101', 'x8_failure_diagnostics_20261102',
               'chemTable_temporal_550_575/final_test_600',
               'highres_snapshot563_lookup_v11/scripts', 'chemTable_v4_full_direct/scripts',
               'chemTable_v4_full_direct/despotic_runtime', 'highres_snapshot575_temporal'):
    sys.path.append(str(ISO/folder))
import common as design

MODELS = design.MODELS
REGIONS = design.REGIONS
BRANCHES = design.BRANCHES
SCENARIOS = (('fiducial', 1., 1., 1.), ('UV0.3', .3, 1., 1.), ('UV3', 3., 1., 1.),
             ('CR0.2', 1., .2, 1.), ('CR5', 1., 5., 1.),
             ('O0.5', 1., 1., .5), ('O2', 1., 1., 2.))


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    design.save(Path(path), value)


def verify():
    manifest = json.loads((ROOT/'manifest.json').read_text())
    for path, sha in manifest['hashes'].items():
        assert digest(path) == sha, path
    for entry in json.loads((ROOT/'inventory.json').read_text()):
        for item in entry['files']:
            st = Path(item['path']).stat()
            assert (st.st_size, st.st_mtime_ns) == (item['size'], item['mtime_ns']), item['path']
    return digest(ROOT/'manifest.json')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def cross(budget, pattern):
    out = budget.copy()
    for a, b in ((3, 4), (5, 6)):
        total = budget[:, a]+budget[:, b]
        fraction = pattern[:, b]/(pattern[:, a]+pattern[:, b])
        out[:, b] = total*fraction
        out[:, a] = total-out[:, b]
        np.testing.assert_allclose(out[:, a]+out[:, b], total, rtol=1e-14)
    return out


def sample_data():
    import h5py
    with h5py.File(SOURCE/'sample.h5') as f:
        data = {k: f['sample'][k][:] for k in ('actual', 'slug', 'mass', 'nuclei', 'stratum',
                                            'inclusion_probability', 'lum_per_H')}
        population = f['stratum_population'][:]
    geometry = np.zeros((len(data['mass']), 9))
    for part in range(4):
        with np.load(SOURCE/f'geometry/part{part:02d}.npz') as f:
            geometry[f['ids']] = f['kernel_weights']
    return data, population, geometry


def log_ratio_and_score(lines, weight, mask):
    """HT signed totals and linearization; mask never converts failure to zero."""
    safe = np.zeros_like(lines)
    safe[mask] = lines[mask]
    contribution = weight[:, None]*safe
    total = contribution.sum(axis=0)
    if np.any(total <= 0) or not np.all(np.isfinite(total)):
        return None, None, total
    point = np.log10(total[0]/total[1])
    score = (contribution[:, 0]/total[0]-contribution[:, 1]/total[1])/np.log(10)
    return point, score, total


def interval(point, score, strata, population):
    if point is None:
        return dict(point=None, standard_error=None, interval_95=None)
    variance = 0.
    for h in np.unique(strata):
        group = score[strata == h]
        n, N = len(group), population[h]
        if n < N:
            assert n >= 2
            variance += n*(1-n/N)*np.var(group, ddof=1)
    se = float(np.sqrt(max(0., variance)))
    return dict(point=float(point), standard_error=se,
                interval_95=[float(point-1.96*se), float(point+1.96*se)])


def nested_draw(strata, count, rng):
    sizes = np.bincount(strata, minlength=270)
    allocation = design.allocation(sizes, count)
    indices = np.sort(np.concatenate([rng.choice(np.flatnonzero(strata == h), int(n), replace=False)
                                     for h, n in enumerate(allocation) if n]))
    return indices, allocation
