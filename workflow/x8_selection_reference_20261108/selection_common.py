"""Frozen D1000-targeted cohort; reuse states, not probability weights."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
ISO=ROOT.parent
OLD=ISO/'x8_aperture_validation_20261105'
PILOT=ISO/'x8_aperture_precision_20261107'
CATALOGUE=ISO/'x8_selection_catalogue_20261107'
sys.path.append(str(OLD))
import aperture_common as parent

base=parent.base
sha=parent.sha
save=parent.save
MODELS=parent.MODELS
MODEL_BRANCH=parent.MODEL_BRANCH
BRANCHES=parent.BRANCHES
DEPTHS=parent.DEPTHS
PARTITIONS=64
APERTURES=606
TOP=64
TAIL=1024
SEED=202611085751000


def verify():
    m=json.loads((ROOT/'manifest.json').read_text())
    for path,digest in m['hashes'].items():
        assert sha(path)==digest,path
    for path,entry in m['frozen_inputs'].items():
        st=Path(path).stat()
        assert (st.st_size,st.st_mtime_ns)==(entry['size'],entry['mtime_ns']),path
    return sha(ROOT/'manifest.json')


def ready():
    manifest=verify();r=json.loads((ROOT/'prepared.json').read_text())
    assert r['complete'] and r['manifest_sha256']==manifest
    assert sha(ROOT/'sample.h5')==r['sample_sha256']
    return manifest,r


def known_models(status,qualified):
    import numpy as np
    tried=status!=0;controls=[0,2,3,4]
    common=tried[:,controls].all(axis=1)|np.any(tried[:,controls]&(~qualified[:,controls]),axis=1)
    return np.column_stack((tried[:,0],tried[:,1],*[common]*4))


def needed_states(status,qualified):
    import numpy as np
    known=known_models(status,qualified)
    need=np.zeros_like(qualified)
    need[:,0]=~known[:,0];need[:,1]=~known[:,1]
    need[:,[0,2,3,4]]|=(~known[:,2])[:,None]
    return need&(status==0)
