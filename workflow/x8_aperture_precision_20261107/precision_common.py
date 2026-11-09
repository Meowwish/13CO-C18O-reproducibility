"""Two-stage direct aperture reference with explicit remaining-particle error."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
ISO = ROOT.parent
PARENT = ISO/'x8_aperture_validation_20261105'
sys.path.append(str(PARENT))
import aperture_common as parent

base = parent.base
sha = parent.sha
save = parent.save
MODELS = parent.MODELS
MODEL_BRANCH = parent.MODEL_BRANCH
BRANCHES = parent.BRANCHES
DEPTHS = parent.DEPTHS
PARTITIONS = 32
SEED = 20261107575125
TOP = 64
TAIL = 1024


def verify():
    manifest = json.loads((ROOT/'manifest.json').read_text())
    for path, digest in manifest['hashes'].items():
        assert sha(path) == digest, path
    for path, entry in manifest['checkpoint_snapshots'].items():
        stat = Path(path).stat()
        assert stat.st_size == entry['size'] and stat.st_mtime_ns == entry['mtime_ns'], path
    return sha(ROOT/'manifest.json')


def ready():
    manifest = verify()
    record = json.loads((ROOT/'prepared.json').read_text())
    assert record['complete'] and record['manifest_sha256'] == manifest
    assert sha(ROOT/'sample.h5') == record['sample_sha256']
    return manifest, record
