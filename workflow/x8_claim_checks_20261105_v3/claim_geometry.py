"""Reuse the validated lattice adjoints on the original independent sample."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import fcntl
import multiprocessing
import os

import h5py
import numpy as np

import claim_common as c
from sfr_geometry import project

WEIGHTS=None


def initialize():
    global WEIGHTS
    import process_snapshot563_lookup_v11
    WEIGHTS=np.load(c.OUTER/'adjoint_weights.npy',mmap_mode='r')[2:4]


def one(task):
    i,xyz,h=task
    field=project(np.asarray(xyz)[None,:],np.array([h])).ravel()
    nz=np.flatnonzero(field)
    assert np.all(field[nz]>=0)
    return i,(WEIGHTS.reshape(2*len(c.REGIONS),-1)[:,nz]@field[nz]).reshape(2,len(c.REGIONS))


def main():
    assert int(os.environ.get('PBS_NCPUS','0'))>0
    sha,_=c.ready();part=int(os.environ['PARTITION']);assert 0<=part<4
    path=c.ROOT/f'geometry/part{part:02d}.npz'
    assert not path.exists()
    lock=(c.ROOT/f'geometry/part{part:02d}.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with h5py.File(c.SOURCE/'sample.h5') as f:xyz,h=f['sample/xyz'][:],f['sample/h'][:]
    ids=np.arange(part,len(h),4);result=np.full((len(ids),2,len(c.REGIONS)),np.nan)
    lookup={int(i):j for j,i in enumerate(ids)}
    with ProcessPoolExecutor(max_workers=min(8,int(os.environ['PBS_NCPUS'])),
            mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
        # Check a real particle in this PBS environment before bulk dispatch.
        first,value=pool.submit(one,(int(ids[0]),xyz[ids[0]],float(h[ids[0]]))).result()
        assert first==int(ids[0]) and np.all(np.isfinite(value)) and np.all(value>=0)
        print('Compute-node projection smoke check passed.',flush=True)
        futures=[pool.submit(one,(int(i),xyz[i],float(h[i]))) for i in ids]
        for count,future in enumerate(as_completed(futures),1):
            i,value=future.result();result[lookup[i]]=value
            if count%512==0:print('claim geometry',part,count,'/',len(ids),flush=True)
    assert np.all(np.isfinite(result)) and np.all(result>=0)
    c.base.atomic_npz(path,ids=ids,kernel=result,complete=np.asarray(True),manifest_sha256=np.asarray(sha))
    lock.close()


if __name__=='__main__':main()
