"""Four disjoint projection partitions; no chemistry is called here."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing
import os

import h5py
import numpy as np

import outer_common as c
from sfr_geometry import project

WEIGHTS = None


def initialize():
    global WEIGHTS
    WEIGHTS = np.load(c.ROOT/'adjoint_weights.npy', mmap_mode='r')


def one(task):
    i, xyz, h = task
    field = project(np.asarray(xyz)[None, :], np.array([h]))
    flat = field.ravel()
    nz = np.flatnonzero(flat)
    assert np.all(flat[nz] >= 0)
    value = (WEIGHTS.reshape(4*len(c.REGIONS), -1)[:, nz]@flat[nz]).reshape(4, len(c.REGIONS))
    return i, value


def main():
    assert int(os.environ.get('PBS_NCPUS', '0')) > 0
    sha, ready = c.ready()
    gr = json.loads((c.ROOT/'geometry_ready.json').read_text())
    assert gr['complete'] and gr['manifest_sha256'] == sha
    assert c.digest(c.ROOT/'adjoint_weights.npy') == gr['weights_sha256']
    part = int(os.environ['PARTITION'])
    assert 0 <= part < 4
    target = c.ROOT/f'geometry/part{part:02d}.npz'
    lock = (c.ROOT/f'geometry/part{part:02d}.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not target.exists()
    with h5py.File(c.ROOT/'sample.h5') as f:
        xyz, h = f['sample/xyz'][:], f['sample/h'][:]
    ids = np.arange(part, len(h), 4)
    result = np.full((len(ids), 4, len(c.REGIONS)), np.nan)
    location = {int(i): j for j, i in enumerate(ids)}
    with ProcessPoolExecutor(max_workers=min(8, int(os.environ['PBS_NCPUS'])),
            mp_context=multiprocessing.get_context('spawn'), initializer=initialize) as pool:
        futures = [pool.submit(one, (int(i), xyz[i], float(h[i]))) for i in ids]
        for done, future in enumerate(as_completed(futures), 1):
            i, value = future.result()
            result[location[i]] = value
            if done % 512 == 0:
                print('outer geometry', part, done, '/', len(ids), flush=True)
    assert np.all(np.isfinite(result)) and np.all(result >= 0)
    c.base.atomic_npz(target, manifest_sha256=np.asarray(sha), sample_sha256=np.asarray(ready['sample_sha256']),
        ids=ids, kernel=result, complete=np.asarray(True))
    lock.close()


if __name__ == '__main__':
    main()
