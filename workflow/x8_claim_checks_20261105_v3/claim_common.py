"""Frozen, cached-reference checks of the paper's central spatial claims."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
ISO = ROOT.parent
SOURCE = ISO/'x8_bounded_reference_20261101'
OUTER = ISO/'x8_outer_validation_20261105'
SFR = ISO/'x8_sfr_validation_20261102'
for folder in (SOURCE, ISO/'x8_full_direct_575_20261028', ISO/'x8_reference_recovery_20261101',
               ISO/'x8_failure_diagnostics_20261102', SFR, OUTER,
               ISO/'chemTable_temporal_550_575/final_test_600',
               ISO/'highres_snapshot575_temporal', ISO/'highres_snapshot563_lookup_v11/scripts',
               ISO/'chemTable_v4_full_direct/despotic_runtime', ISO/'chemTable_v4_full_direct/scripts', ISO):
    sys.path.append(str(folder))
import common as base
import sfr_common as sfr

MODELS = base.MODELS
MODEL_BRANCH = base.MODEL_BRANCH
REGIONS = tuple([f'annulus_{r}_{r+1}' for r in range(15)]+
                ['inner_1_8','outer_8_15','disc_0_15','full_field'])
REFERENCES = (('default', ISO/'x8_reference_recovery_20261101/candidate_overlay.npz'),
              ('extended', ISO/'x8_failure_diagnostics_20261102/diagnostic_candidates.npz'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    base.save(path, value)


def verify():
    m = json.loads((ROOT/'manifest.json').read_text())
    for path, digest in m['hashes'].items():
        assert sha(path)==digest, path
    return sha(ROOT/'manifest.json')


def ready():
    sha_m = verify()
    r = json.loads((ROOT/'prepared.json').read_text())
    assert r['complete'] and r['manifest_sha256']==sha_m
    for path, digest in r['hashes'].items():
        assert sha(path)==digest, path
    return sha_m, r


def estimate(values, prediction, mask, kernel, weight):
    """Log ratios and particle influence scores, retaining numerical nonresponse."""
    d = np.where(mask[:,None], values, 0.)
    p = np.where(mask[:,None], prediction, 0.)
    wk = kernel*weight[:,None]
    D, P = wk.T@d, wk.T@p
    good = np.all(D>0,axis=1)&np.all(P>0,axis=1)
    direct = np.full(len(D), np.nan); proxy = direct.copy()
    sd = np.zeros_like(wk); sp = np.zeros_like(wk)
    direct[good] = np.log10(D[good,0]/D[good,1])
    proxy[good] = np.log10(P[good,0]/P[good,1])
    sd[:,good] = wk[:,good]*(d[:,0,None]/D[None,good,0]-d[:,1,None]/D[None,good,1])/np.log(10)
    sp[:,good] = wk[:,good]*(p[:,0,None]/P[None,good,0]-p[:,1,None]/P[None,good,1])/np.log(10)
    all_light = wk.T@np.maximum(prediction,0.)
    missing = (wk*(~mask)[:,None]).T@np.maximum(prediction,0.)
    missing = np.divide(missing,all_light,out=np.full_like(missing,np.nan),where=all_light>0)
    return dict(direct=direct, proxy=proxy, direct_score=sd, proxy_score=sp,
                missing_predicted_positive_fraction=missing)


def contrast(m, coefficient, design):
    use = coefficient!=0
    d = float(coefficient[use]@m['direct'][use])
    p = float(coefficient[use]@m['proxy'][use])
    sd = m['direct_score']@coefficient
    sp = m['proxy_score']@coefficient
    return dict(direct=sfr.interval(d,sd,design), matched_surrogate=p,
                direct_minus_surrogate=sfr.interval(d-p,sd-sp,design)), sd, sp


def spatial_slope_replicas(coefficient, leave):
    """Contract bin coefficients, keeping sector, bin and model axes explicit."""
    use=coefficient!=0
    selected=leave[:,use]
    assert selected.ndim==4 and selected.shape[1]==use.sum() and selected.shape[-1]==2
    assert np.all(selected>0)
    ratios=np.log10(selected[...,0]/selected[...,1])
    result=np.einsum('b,sbm->sm',coefficient[use],ratios)
    assert result.shape==(len(leave),leave.shape[2])
    return result
