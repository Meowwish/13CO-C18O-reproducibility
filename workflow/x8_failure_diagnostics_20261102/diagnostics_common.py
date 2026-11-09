"""Local outputs and frozen parent lineage; no DESPOTIC solve on import."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
ISO = ROOT.parent
SOURCE = ISO / 'x8_bounded_reference_20261101'
PARENT = ISO / 'x8_reference_recovery_20261101'
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(PARENT))
import common as base
import shared as parent

PROTOCOLS = ('default_600s', 'mxstep5000_600s')
WALL_SECONDS = 600
MXSTEP = 5000
SEED = 20261102575


def save(path, value):
    base.save(path, value)


def verify():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for path, sha in manifest['hashes'].items():
        assert base.core.digest(path) == sha, path
    assert parent.verify() == manifest['parent_manifest_sha256']
    return base.core.digest(ROOT / 'manifest.json')


def importance(proxy, nuclei, pi, geometry):
    """Largest weighted rare-line fraction, all six models and nine regions."""
    score = np.zeros(proxy.shape[:2])
    for g in geometry:
        for r in range(g.shape[1]):
            weighted = nuclei[:, None, None] / pi[:, None, None] * g[:, r, None, None] * np.maximum(proxy[:, :, 1:3], 0)
            denominator = weighted.sum(axis=0)
            fraction = np.divide(weighted, denominator[None], out=np.zeros_like(weighted), where=denominator[None] > 0)
            score = np.maximum(score, fraction.max(axis=2))
    # Actual/on is shared by the production and paired actual/on models.
    return np.column_stack((np.maximum(score[:, 0], score[:, 2]), score[:, 1], score[:, 3:6]))


def admissible(status, final):
    return (status == 1) & np.all(final == 1, axis=-1)


def agreement_error(old, new):
    """dex for positive pairs, signed relative difference otherwise."""
    old, new = np.asarray(old), np.asarray(new)
    positive = (old > 0) & (new > 0)
    error = np.abs(new-old) / np.maximum(np.maximum(np.abs(old), np.abs(new)), 1e-60)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(positive, np.abs(np.log10(new/old)), error)
