"""Exact adjoint of the existing finite-normalised Gaussian beam operator."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import fcntl
import json
import multiprocessing
import os
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
import common as c

WEIGHTS=None
PROD=None
SETTINGS=None


def settings():
    return json.loads((c.ISO/'highres_snapshot575_temporal/config.json').read_text())


def adjoint(masks,sigma,pixel):
    denominator=gaussian_filter(np.ones(masks.shape[1:]),sigma,mode='constant',cval=0)
    return np.stack([gaussian_filter(m/denominator,sigma,mode='constant',cval=0)*pixel**2 for m in masks])


def prepare_weights():
    cfg=settings()['projection'];res=cfg['map_resolution'];size=cfg['map_size_pc'];pixel=size/res
    xy=(np.arange(res)+.5)*pixel-size/2;x,y=np.meshgrid(xy,xy,indexing='ij')
    masks=c.region_masks(np.column_stack((x.ravel(),y.ravel()))).T.reshape(9,res,res)
    weights=adjoint(masks,125/(np.sqrt(8*np.log(2))*pixel),pixel)
    path=c.ROOT/'adjoint_weights.npy';assert not path.exists()
    with path.open('xb') as f:np.save(f,weights)
    return weights


def project(xyz,h):
    import process_snapshot563_lookup_v11 as prod
    cfg=settings()['projection']
    return prod.project_surface(np.asarray(xyz),np.asarray(h),np.ones(len(h)),cfg['map_size_pc'],cfg['map_resolution'])


def integrate(field,weights):
    flat=field.ravel();nz=np.flatnonzero(flat)
    assert np.all(flat[nz]>=0)
    return weights.reshape(9,-1)[:,nz]@flat[nz]


def check_forward(selected,weights):
    cfg=settings()['projection'];size=cfg['map_size_pc'];res=cfg['map_resolution'];pixel=size/res
    xy=(np.arange(res)+.5)*pixel-size/2;x,y=np.meshgrid(xy,xy,indexing='ij')
    masks=c.region_masks(np.column_stack((x.ravel(),y.ravel()))).T.reshape(9,res,res)
    sigma=125/(np.sqrt(8*np.log(2))*pixel)
    denominator=gaussian_filter(np.ones((res,res)),sigma,mode='constant',cval=0)
    picks=np.unique([0,len(selected['h'])//2,len(selected['h'])-1,int(np.argmax(selected['h']))])
    maximum=0.
    for i in picks:
        field=project(selected['xyz'][i:i+1],selected['h'][i:i+1])
        beam=gaussian_filter(field,sigma,mode='constant',cval=0)/denominator*pixel**2
        forward=np.asarray([beam[m].sum() for m in masks]);back=integrate(field,weights)
        np.testing.assert_allclose(back,forward,rtol=2e-10,atol=1e-10)
        maximum=max(maximum,float(np.max(np.abs(back-forward))))
    with h5py.File(c.PILOT/'pilot_sample.h5') as f:
        for i in (0,17,41,73,101,127):
            field=project(f['unique/xyz'][i:i+1],f['unique/h'][i:i+1])
            np.testing.assert_allclose(integrate(field,weights),f['unique/kernel_weights'][i],rtol=2e-5,atol=1e-8)
    return dict(passed=True,forward_cases=len(picks),old_pilot_cases=6,max_forward_absolute_difference=maximum)


def initialize():
    global WEIGHTS
    WEIGHTS=np.load(c.ROOT/'adjoint_weights.npy',mmap_mode='r')


def one(task):
    i,xyz,h=task
    return i,integrate(project(np.asarray(xyz)[None,:],np.asarray([h])),WEIGHTS)


def main():
    assert int(os.environ.get('PBS_NCPUS','0'))>0
    sha,r=c.ready();part=int(os.environ['GEOMETRY_PART']);assert 0<=part<4
    assert c.core.digest(c.ROOT/'adjoint_weights.npy')==r['geometry_weights_sha256']
    with h5py.File(c.ROOT/'sample.h5') as f:
        xyz=f['sample/xyz'][:];h=f['sample/h'][:]
    ids=np.arange(part,len(h),4);target=c.ROOT/f'geometry/part{part:02d}.npz'
    lock=(c.ROOT/f'geometry/part{part:02d}.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not target.exists()
    result=np.full((len(ids),9),np.nan);lookup={int(i):j for j,i in enumerate(ids)}
    with ProcessPoolExecutor(max_workers=min(8,int(os.environ['PBS_NCPUS'])),
            mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
        futures=[pool.submit(one,(int(i),xyz[i],float(h[i]))) for i in ids]
        for k,future in enumerate(as_completed(futures),1):
            i,value=future.result();result[lookup[i]]=value
            if k%512==0:print('geometry',part,k,'/',len(ids),flush=True)
    assert np.all(np.isfinite(result)&(result>=0)) and np.all(result[:,0]>0)
    c.atomic_npz(target,manifest_sha256=np.asarray(sha),sample_sha256=np.asarray(r['sample_sha256']),
        ids=ids,kernel_weights=result,complete=np.asarray(True))
    lock.close()


if __name__=='__main__':main()
