"""Exact adjoint for bilinear sampling after the finite Gaussian beam."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing
import os

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter

import sfr_common as c

WEIGHTS = None


def splat(mask, xy_pixel, shape):
    coordinates = np.clip(xy_pixel, 0., np.array(shape)-1.)
    lo = np.floor(coordinates).astype(int)
    hi = np.minimum(lo+1, np.array(shape)-1)
    fraction = coordinates-lo
    result = np.zeros(shape)
    for ix in (0, 1):
        for iy in (0, 1):
            x = hi[:, 0] if ix else lo[:, 0]
            y = hi[:, 1] if iy else lo[:, 1]
            weight = (fraction[:, 0] if ix else 1-fraction[:, 0])*(fraction[:, 1] if iy else 1-fraction[:, 1])
            np.add.at(result, (x, y), np.asarray(mask)*weight)
    return result


def adjoint(mask, xy_pixel, shape, sigma):
    denominator = gaussian_filter(np.ones(shape), sigma, mode='constant', cval=0., truncate=4.)
    return gaussian_filter(splat(mask, xy_pixel, shape)/denominator, sigma, mode='constant', cval=0., truncate=4.)


def project(xyz, h):
    import process_snapshot563_lookup_v11 as production
    return production.project_surface(np.asarray(xyz), np.asarray(h), np.ones(len(h)), 30000., 1500)


def integrate(field, weights):
    flat = field.ravel()
    nz = np.flatnonzero(flat)
    assert np.all(flat[nz] >= 0)
    return weights.reshape(len(weights), -1)[:, nz] @ flat[nz]


def initialize():
    global WEIGHTS
    WEIGHTS = np.load(c.ROOT/'adjoint_weights.npy', mmap_mode='r')


def one(task):
    i, xyz, h = task
    return i, integrate(project(np.asarray(xyz)[None, :], np.array([h])), WEIGHTS)


def main():
    assert int(os.environ.get('PBS_NCPUS', '0')) > 0
    sha, ready = c.ready()
    part = int(os.environ['GEOMETRY_PART'])
    assert 0 <= part < 4
    target = c.ROOT/f'geometry/part{part:02d}.npz'
    lock = (c.ROOT/f'geometry/part{part:02d}.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not target.exists()
    with h5py.File(c.SOURCE/'sample.h5') as f:
        xyz, h = f['sample/xyz'][:], f['sample/h'][:]
    ids = np.arange(part, len(h), 4)
    result = np.full((len(ids), ready['bin_count']), np.nan)
    lookup = {int(i): j for j, i in enumerate(ids)}
    with ProcessPoolExecutor(max_workers=min(8, int(os.environ['PBS_NCPUS'])),
            mp_context=multiprocessing.get_context('spawn'), initializer=initialize) as pool:
        futures = [pool.submit(one, (int(i), xyz[i], float(h[i]))) for i in ids]
        for count, future in enumerate(as_completed(futures), 1):
            i, value = future.result()
            result[lookup[i]] = value
            if count % 512 == 0:
                print('SFR geometry', part, count, '/', len(ids), flush=True)
    assert np.all(np.isfinite(result)) and np.all(result >= 0)
    c.base.atomic_npz(target, manifest_sha256=np.asarray(sha), sample_sha256=np.asarray(ready['sample_sha256']),
                      catalogue_sha256=np.asarray(ready['catalogue_sha256']), ids=ids, kernel=result, complete=np.asarray(True))
    lock.close()


if __name__ == '__main__':
    main()
