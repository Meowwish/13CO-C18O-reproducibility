"""Independent outer-disc validation; frozen models and explicit sample design."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
ISO = ROOT.parent
SOURCE = ISO/'x8_bounded_reference_20261101'
PARENT = ISO/'x8_failure_diagnostics_20261102'
FULL = ISO/'x8_full_direct_575_20261028'
for folder in (FULL, SOURCE, ISO/'x8_reference_recovery_20261101', PARENT,
               ISO/'highres_snapshot575_temporal', ISO):
    sys.path.insert(0, str(folder))
import common as base
import diagnostics_common as diagnostic
sys.path.insert(0, str(ROOT))

SEED = 20261105575
PARTITIONS = 8
RADIAL_TARGETS = np.array([256]*8+[4096]*7+[2048])
STRATA_PER_RADIUS = 30
MODELS = base.MODELS
BRANCHES = base.BRANCHES
MODEL_BRANCH = base.MODEL_BRANCH
REGIONS = tuple([f'annulus_{r}_{r+1}' for r in range(15)]+
                ['inner_1_8', 'outer_8_15', 'disc_0_15', 'full_field'])
OPERATORS = ('native_normalized', 'native_zero_padded',
             'lattice_normalized', 'lattice_zero_padded')
MAPS = [ISO/'highres_snapshot575_temporal/output/snapshot_575_maps_125pc.npz',
        ISO/'highres_snapshot575_temporal/slug_only/output/snapshot_575_maps_125pc.npz',
        ISO/'isotope_fractionation_controls/x8_snapshot575/maps_125pc.npz']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    base.save(path, value)


def verify():
    m = json.loads((ROOT/'manifest.json').read_text())
    for path, sha in m['hashes'].items():
        assert digest(path) == sha, path
    assert diagnostic.verify() == m['parent_manifest_sha256']
    return digest(ROOT/'manifest.json')


def ready():
    sha = verify()
    r = json.loads((ROOT/'sample_ready.json').read_text())
    assert r['complete'] and r['manifest_sha256'] == sha
    assert digest(ROOT/'sample.h5') == r['sample_sha256']
    return sha, r


def strata(data, ratios, totals):
    r = np.digitize(np.hypot(*data['xyz'][:, :2].T)/1000., np.arange(1, 16))
    density = np.digitize(np.log10(data['actual'][:, 0]), [1, 3])
    fraction = data['nuclei'][:, None, None]*data['lum_per_H'][:, :, 1:3]/totals[None, :, 1:3]
    brightness = np.digitize(np.log10(np.maximum(fraction.max(axis=(1, 2)), 1e-300)), base.LUMINOSITY_EDGES)
    carbon = data['actual'][:, 3]/np.maximum(data['actual'][:, 4], 1e-30)/ratios[0]
    oxygen = data['actual'][:, 5]/np.maximum(data['actual'][:, 6], 1e-30)/ratios[1]
    edge = (~np.all(data['covered'], axis=1)) | (carbon < .5) | (carbon > 2) | (oxygen < .5) | (oxygen > 2)
    return ((r*3+density)*5+brightness)*2+edge.astype('i8')


def allocate(population):
    population = np.asarray(population)
    result = np.zeros_like(population)
    for radial, target in enumerate(RADIAL_TARGETS):
        sl = slice(radial*STRATA_PER_RADIUS, (radial+1)*STRATA_PER_RADIUS)
        available = population[sl]
        result[sl] = base.allocation(available, min(int(target), int(available.sum())))
    assert np.all(result <= population)
    assert np.all((population == 0) | (result >= np.minimum(population, 2)))
    return result


def masks(radius):
    return [(radius >= r) & (radius < r+1) for r in range(15)]+[
        (radius >= 1) & (radius < 8), (radius >= 8) & (radius < 15),
        radius < 15, np.ones_like(radius, dtype=bool)]


def covariance(scores, strata_ids, population, allocation):
    from design_statistics import joint_covariance
    return joint_covariance(scores, strata_ids, population, allocation)


def interval(point, variance, z=1.96):
    if not np.isfinite(point) or not np.isfinite(variance):
        return dict(point=None, standard_error=None, interval_95=None)
    se = float(np.sqrt(max(0., variance)))
    return dict(point=float(point), standard_error=se,
                interval_95=[float(point-z*se), float(point+z*se)])
