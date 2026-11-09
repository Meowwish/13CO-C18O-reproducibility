"""Frozen follow-up experiments; no numerical solve on import."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
ISO = ROOT.parent
for name in ('x8_aperture_validation_20261105', 'x8_full_direct_575_20261028',
             'x8_bounded_reference_20261101', 'x8_reference_recovery_20261101',
             'x8_failure_diagnostics_20261102', 'x8_sfr_validation_20261102',
             'x8_outer_validation_20261105', 'chemTable_temporal_550_575/final_test_600',
             'highres_snapshot563_lookup_v11/scripts', 'chemTable_v4_full_direct/scripts',
             'chemTable_v4_full_direct/despotic_runtime', 'highres_snapshot575_temporal'):
    sys.path.append(str(ISO/name))
sys.path.append(str(ISO))
import aperture_common as parent

MODELS = parent.MODELS
BRANCHES = parent.BRANCHES
MAP_BRANCH = parent.MODEL_BRANCH
SNAPSHOTS = (551, 563, 567)
PARTITIONS = 16
TARGETS = np.array([1024]*8 + [3072]*8)
SEED = 20261109575


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    parent.base.save(path, value)


def verify():
    d = json.loads((ROOT/'manifest.json').read_text())
    for path, digest in d['hashes'].items():
        assert sha(path) == digest, path
    for path, entry in d['large_inputs'].items():
        s = Path(path).stat()
        assert (s.st_size, s.st_mtime_ns) == (entry['size'], entry['mtime_ns']), path
    return sha(ROOT/'manifest.json')


def srs_variance(scores, strata, population):
    x = np.asarray(scores)
    v = np.zeros(x.shape[1:])
    for h in np.unique(strata):
        row = x[strata == h]; n = len(row); N = population[h]
        if n < N:
            assert n >= 2
            v += n*(1-n/N)*np.var(row, axis=0, ddof=1)
    return v


def estimate(lines, weight, mask, strata, population):
    z = np.where(mask[:, None], lines, 0.)*weight[:, None]
    total = z.sum(axis=0)
    if np.any(total <= 0):
        return None
    point = np.log10(total[0]/total[1])
    score = (z[:, 0]/total[0]-z[:, 1]/total[1])/np.log(10)
    return point, score


def result(point, score, strata, population):
    se = float(np.sqrt(srs_variance(score, strata, population)))
    return dict(point=float(point), standard_error=se,
                interval_95=[float(point-1.96*se), float(point+1.96*se)])


def assemble(batch):
    folder = ROOT/batch
    ready = json.loads((folder/'prepared.json').read_text())
    import h5py
    with h5py.File(folder/'cases.h5') as f:
        assert sha(folder/'cases.h5') == ready['cases_sha256']
        n = len(f['raw']); particle = f['particle_id'][:]
    values = np.full((n, 27), np.nan); q = np.zeros(n, bool); status = np.zeros(n, 'i1')
    for part in range(PARTITIONS):
        with np.load(folder/f'direct/part{part:02d}.npz') as f:
            ids = f['case_ids']
            np.testing.assert_array_equal(ids, np.arange(part, n, PARTITIONS))
            np.testing.assert_array_equal(f['particle_id'], particle[ids])
            assert bool(f['complete']) and np.all(f['status'] != 0)
            np.testing.assert_array_equal(f['qualified'], (f['status'] == 1) & np.all(f['final_returns'] == 1, axis=1))
            assert json.loads(str(f['metadata']))['cases_sha256'] == ready['cases_sha256']
            values[ids], q[ids], status[ids] = f['values'], f['qualified'], f['status']
    assert np.all(status != 0)
    return values, q, status
