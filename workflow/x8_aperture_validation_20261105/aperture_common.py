"""Census of particle contributions to a probability sample of apertures."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
ISO = ROOT.parent
for path in ('x8_full_direct_575_20261028', 'x8_bounded_reference_20261101',
             'x8_reference_recovery_20261101', 'x8_failure_diagnostics_20261102',
             'x8_sfr_validation_20261102', 'x8_outer_validation_20261105',
             'chemTable_temporal_550_575/final_test_600', 'highres_snapshot575_temporal',
             'highres_snapshot563_lookup_v11/scripts', 'chemTable_v4_full_direct/scripts',
             'chemTable_v4_full_direct/despotic_runtime'):
    sys.path.append(str(ISO/path))
sys.path.append(str(ISO))
import common as base

MODELS = base.MODELS
BRANCHES = base.BRANCHES
MODEL_BRANCH = base.MODEL_BRANCH
PARTITIONS = 32
SEED = 20261105575125
DEPTHS = 10.**np.arange(7)
MAPS = [ISO/'highres_snapshot575_temporal/output/snapshot_575_maps_125pc.npz',
        ISO/'highres_snapshot575_temporal/slug_only/output/snapshot_575_maps_125pc.npz',
        ISO/'isotope_fractionation_controls/x8_snapshot575/maps_125pc.npz']


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    base.save(path, value)


def verify():
    m = json.loads((ROOT/'manifest.json').read_text())
    for path, digest in m['hashes'].items():
        assert sha(path) == digest, path
    registry = json.loads((ISO/'x8_full_direct_575_20261028/registry.json').read_text())
    for item in registry['parts']:
        stat = Path(item['path']).stat()
        assert stat.st_size == item['size'] and stat.st_mtime_ns == item['mtime_ns'], item['path']
    return sha(ROOT/'manifest.json')


def ready():
    manifest = verify()
    r = json.loads((ROOT/'prepared.json').read_text())
    assert r['complete'] and r['manifest_sha256'] == manifest
    assert sha(ROOT/'census.h5') == r['census_sha256']
    return manifest, r


def predict(raws):
    import joblib
    from adaptive_local_surrogate import (AdaptiveLocalActualAbundanceSurrogate, OUTPUT_NAMES,
        transform_outputs, raw_inputs_to_features, inverse_transform_outputs)
    model_path = ISO/'chemTable_temporal_550_575/final_test_600/frozen_surrogate.joblib'
    model = AdaptiveLocalActualAbundanceSurrogate(model_path, require_validated=False)
    bundle = joblib.load(ISO/'isotope_fractionation_controls/intervention_models/residual_leaves15.joblib')
    assert bundle['base_sha256'] == sha(model_path)
    predictions = []
    for x in raws[:2]:
        v, _ = model.query(x, chunk_size=100000)
        predictions.append(np.column_stack([v[k] for k in OUTPUT_NAMES]))
    for x in (raws[0], raws[2]):
        v, _ = model.query(x, chunk_size=100000)
        bt = transform_outputs(x, np.column_stack([v[k] for k in OUTPUT_NAMES]))
        aug = np.column_stack((raw_inputs_to_features(x), bt))
        for physics in ('on', 'off'):
            correction = np.column_stack([m.predict(aug) for m in bundle['models'][physics]])
            predictions.append(inverse_transform_outputs(x, bt+correction, bundle['target_clips'][physics]))
    return np.stack(predictions, axis=1)
