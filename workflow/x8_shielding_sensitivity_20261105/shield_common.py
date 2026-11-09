"""Experiment-local lineage and small statistical helpers."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent
ISO=ROOT.parent
SOURCE=ISO/'x8_bounded_reference_20261101'
OUTER=ISO/'x8_outer_validation_20261105'
CLAIMS=ISO/'x8_claim_checks_20261105_v3'
SFR=ISO/'x8_sfr_validation_20261102'
for folder in (SOURCE,ISO/'x8_full_direct_575_20261028',ISO/'x8_reference_recovery_20261101',
        ISO/'x8_failure_diagnostics_20261102',SFR,ISO/'chemTable_v4_full_direct/scripts',
        ISO/'chemTable_v4_full_direct/despotic_runtime',ISO):sys.path.append(str(folder))
import common as base
import sfr_common as stat

SEED=20261105575
PARTITIONS=16
BRANCHES=base.BRANCHES
MODES=('baseline','fixed35','fixed69','molecular')
REGIONS=tuple([f'annulus_{i}_{i+1}' for i in range(15)]+['inner_1_8','outer_8_15','disc_0_15','full_field'])
PAIRS=dict(source=(1,0),exchange=(0,2),uniform_exchange=(3,4),pattern_on=(0,3),pattern_off=(2,4))


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,value):base.save(Path(path),value)


def verify():
    for path,digest in json.loads((ROOT/'manifest.json').read_text())['hashes'].items():assert sha(path)==digest,path
    return sha(ROOT/'manifest.json')


def ready():
    digest=verify();r=json.loads((ROOT/'prepared.json').read_text())
    assert r['complete'] and r['manifest_sha256']==digest
    assert sha(ROOT/'sample.h5')==r['sample_sha256']
    return digest,r


def nested(strata,population,count,rng,radial_targets=None):
    available=np.bincount(strata,minlength=len(population))
    if radial_targets is None:allocation=base.allocation(available,count)
    else:
        allocation=np.zeros_like(available)
        for radial,target in enumerate(radial_targets):
            ids=np.flatnonzero(np.arange(len(available))//30==radial)
            allocation[ids]=base.allocation(available[ids],int(target))
    assert allocation.sum()==count
    assert np.all(allocation[population>0]>0)
    assert np.all((allocation==population)|(allocation>=2))
    ids=np.sort(np.concatenate([rng.choice(np.flatnonzero(strata==h),int(n),replace=False)
        for h,n in enumerate(allocation) if n]))
    return ids,allocation


def ratio(values,mask,kernel,weight):
    safe=np.where(mask[:,None],values,0.)
    wk=weight[:,None]*kernel;totals=wk.T@safe
    good=np.all(np.isfinite(totals)&(totals>0),axis=1)
    result=np.full(len(totals),np.nan);score=np.zeros_like(wk)
    result[good]=np.log10(totals[good,0]/totals[good,1])
    score[:,good]=wk[:,good]*(safe[:,0,None]/totals[None,good,0]-safe[:,1,None]/totals[None,good,1])/np.log(10)
    return result,score
