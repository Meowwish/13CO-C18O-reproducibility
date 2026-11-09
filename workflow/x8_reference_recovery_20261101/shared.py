import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'x8_bounded_reference_20261101'
sys.path.insert(0,str(SOURCE))
import common as base


def save(path,value):base.save(path,value)


def verify():
    r=json.loads((ROOT/'manifest.json').read_text())
    for path,sha in r['hashes'].items():assert base.core.digest(path)==sha,path
    assert base.verify()==r['source_manifest_sha256']
    assert base.core.digest(SOURCE/'sample.h5')==r['source_sample_sha256']
    return base.core.digest(ROOT/'manifest.json')
