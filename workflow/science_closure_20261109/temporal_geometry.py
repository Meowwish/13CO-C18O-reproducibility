"""Reuse time-independent exact adjoints with new particle coordinates."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os

import h5py
import numpy as np

import closure_common as c
from sfr_geometry import project

WEIGHTS = None


def initialize():
    global WEIGHTS
    WEIGHTS = np.load(c.ISO/'x8_outer_validation_20261105/adjoint_weights.npy', mmap_mode='r')


def one(task):
    i, xyz, h = task
    field = project(xyz[None, :], np.array([h])).ravel()
    nz = np.flatnonzero(field)
    out = (WEIGHTS.reshape(76, -1)[:, nz]@field[nz]).reshape(4, 19)
    assert np.isfinite(out).all() and np.all(out >= 0)
    return i, out


def main():
    assert int(os.environ.get('PBS_NCPUS', '0')) > 0
    manifest = c.verify(); snap = int(os.environ['SNAPSHOT']); assert snap in c.SNAPSHOTS
    folder = c.ROOT/f'temporal{snap}'
    old = c.ISO/'x8_outer_validation_20261105'
    assert c.sha(old/'adjoint_weights.npy') == json.loads((old/'geometry_ready.json').read_text())['weights_sha256']
    ready = json.loads((folder/'prepared.json').read_text())
    assert c.sha(folder/'sample.h5') == ready['sample_sha256']
    with h5py.File(folder/'sample.h5') as f:xyz, h = f['xyz'][:], f['h'][:]
    result = np.full((len(h), 4, 19), np.nan)
    with ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context('spawn'), initializer=initialize) as pool:
        for i, value in pool.map(one, [(i, xyz[i], h[i]) for i in range(len(h))], chunksize=16):result[i] = value
    assert np.isfinite(result).all()
    c.parent.base.atomic_npz(folder/'geometry.npz', kernel=result, manifest_sha256=np.asarray(manifest),
        sample_sha256=np.asarray(ready['sample_sha256']), complete=np.asarray(True))


if __name__ == '__main__':main()
