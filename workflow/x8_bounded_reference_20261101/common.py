"""Frozen lineage, design helpers and atomic experiment-local outputs."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
ISO = ROOT.parent
FULL = ISO / 'x8_full_direct_575_20261028'
PILOT = ISO / 'x8_direct_validation_20261027'
sys.path.insert(0, str(FULL))
import core

BRANCHES = core.BRANCHES
MODELS = ('production_full','production_slug','actual_on','actual_off','uniform_on','uniform_off')
MODEL_BRANCH = np.array([0,1,0,2,3,4])
REGIONS = ('full_field','inner_1_8',*[f'annulus_{a}_{a+1}' for a in range(1,8)])
DRAW_COUNT = 32768
SEED = 20261101575
PARTITIONS = 8
LUMINOSITY_EDGES = np.array([-12.,-10.,-8.,-6.])


def clean(value):
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,(bool,np.bool_)): return bool(value)
    if isinstance(value,(int,np.integer)): return int(value)
    if isinstance(value,(float,np.floating)): return float(value) if np.isfinite(value) else None
    return value


def save(path, value): core.save_json(path, clean(value))


def verify():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for path,sha in manifest['hashes'].items(): assert core.digest(path)==sha,path
    sha=core.verify_code()
    assert sha==manifest['original_manifest_sha256']
    audit=json.loads((FULL/'input_audit.json').read_text())
    assert audit['passed'] and audit['manifest_sha256']==sha
    assert core.digest(FULL/'preflight.json')==audit['preflight_sha256']
    return core.digest(ROOT/'manifest.json')


def ready():
    sha=verify();r=json.loads((ROOT/'sample_ready.json').read_text())
    assert r['complete'] and r['manifest_sha256']==sha
    assert core.digest(ROOT/'sample.h5')==r['sample_sha256']
    return sha,r


def atomic_npz(path, **arrays):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    tmp=Path(str(path)+f'.tmp.{os.getpid()}')
    with tmp.open('wb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def allocation(population, count=DRAW_COUNT):
    population=np.asarray(population,dtype='i8');assert population.sum()>=count
    n=np.zeros_like(population)
    while n.sum()<count:
        active=np.flatnonzero(n<population)
        amount=min(len(active),int(count-n.sum()))
        n[active[:amount]]+=1
    assert n.sum()==count and np.all((population==0)|(n>0))
    return n


def design_strata(data, ratios, total):
    radius=np.hypot(*data['xyz'][:,:2].T)/1000
    radial=np.digitize(radius,np.arange(1,9))
    density=np.digitize(np.log10(data['actual'][:,0]),[1,3])
    fractional=data['nuclei'][:,None,None]*data['lum_per_H'][:,:,1:3]/total[None,:,1:3]
    importance=np.max(fractional,axis=(1,2))
    bright=np.digitize(np.log10(np.maximum(importance,1e-300)),LUMINOSITY_EDGES)
    raw=data['actual']
    cr=raw[:,3]/np.maximum(raw[:,4],1e-30)/ratios[0]
    ore=raw[:,5]/np.maximum(raw[:,6],1e-30)/ratios[1]
    edge=(~np.all(data['covered'],axis=1))|(cr<.5)|(cr>2)|(ore<.5)|(ore>2)
    return ((radial*3+density)*5+bright)*2+edge.astype('i8')


def uniform(raw,ratios):
    out=raw.copy()
    for a,b,r in ((3,4,ratios[0]),(5,6,ratios[1])):
        total=raw[:,a]+raw[:,b];out[:,b]=total/(1+r);out[:,a]=total-out[:,b]
        np.testing.assert_allclose(out[:,a]+out[:,b],total,rtol=1e-14)
    return out


def region_masks(xyz):
    r=np.hypot(*xyz[:,:2].T)/1000
    return np.column_stack((np.ones(len(r),bool),(r>=1)&(r<8),*[(r>=a)&(r<a+1) for a in range(1,8)]))
