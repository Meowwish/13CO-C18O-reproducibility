"""Freeze exact adjoints for both published map boundary operators."""
import json
import os

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter

import outer_common as c
from plot_galic_selection import lattice, sample
from sfr_geometry import splat, project


def adjoint(mask, sigma, normalized):
    if normalized:
        denominator = gaussian_filter(np.ones(mask.shape), sigma, mode='constant', cval=0., truncate=4.)
        mask = mask/denominator
    return gaussian_filter(mask, sigma, mode='constant', cval=0., truncate=4.)


def main():
    assert int(os.environ.get('PBS_NCPUS', '0')) > 0
    sha, ready = c.ready()
    assert not (c.ROOT/'geometry_ready.json').exists()
    xy = lattice()
    lattice_masks = c.masks(np.hypot(*xy.T))
    pixels = (xy+15)/.02-.5
    axis = (np.arange(1500)+.5)*.02-15
    native_radius = np.hypot(axis[:, None], axis[None, :])
    native_masks = c.masks(native_radius)
    sigma = 125/(np.sqrt(8*np.log(2))*20)
    weights = np.lib.format.open_memmap(c.ROOT/'adjoint_weights.npy', mode='w+',
        dtype='f8', shape=(4, len(c.REGIONS), 1500, 1500))
    for op in range(4):
        for r in range(len(c.REGIONS)):
            mask = native_masks[r].astype(float)*400 if op < 2 else splat(lattice_masks[r], pixels, (1500, 1500))
            weights[op, r] = adjoint(mask, sigma, op % 2 == 0)
        print('prepared', c.OPERATORS[op], flush=True)
    weights.flush()
    # Catalogue exact cached surrogate curves without any new chemical solve.
    exact = np.empty((2, len(c.REGIONS), 6, 2))
    for model, name in enumerate(c.MODELS):
        path = c.MAPS[model] if model < 2 else c.MAPS[2]
        prefix = '' if model < 2 else name+'_'
        with np.load(path) as f:
            for line, key in enumerate(('W13CO', 'WC18O')):
                field = f[prefix+key]
                assert field.shape == (1500, 1500) and np.all(np.isfinite(field)) and np.all(field >= 0)
                for r, mask in enumerate(native_masks):
                    exact[0, r, model, line] = field[mask].sum()*400
                sampled = sample(field, xy)
                for r, mask in enumerate(lattice_masks):
                    exact[1, r, model, line] = sampled[mask].sum()
    # Verify every operator against actual forward particle kernels, including
    # the sample particles nearest the finite field edges.
    with h5py.File(c.ROOT/'sample.h5') as f:
        xyz, h = f['sample/xyz'][:], f['sample/h'][:]
    picks = np.unique([0, len(h)//2, len(h)-1, int(np.argmax(h)),
                       int(np.argmin(np.abs(np.max(np.abs(xyz[:, :2]), axis=1)-15000)))])
    denominator = gaussian_filter(np.ones((1500, 1500)), sigma, mode='constant', cval=0.)
    max_error = 0.
    for i in picks:
        field = project(xyz[i:i+1], h[i:i+1])
        smooth = gaussian_filter(field, sigma, mode='constant', cval=0.)
        flat = field.ravel()
        nz = np.flatnonzero(flat)
        for op in range(4):
            beam = smooth/denominator if op % 2 == 0 else smooth
            if op < 2:
                forward = np.array([beam[m].sum()*400 for m in native_masks])
            else:
                sampled = sample(beam, xy)
                forward = np.array([sampled[m].sum() for m in lattice_masks])
            back = weights[op].reshape(len(c.REGIONS), -1)[:, nz]@flat[nz]
            np.testing.assert_allclose(back, forward, rtol=3e-10, atol=1e-12)
            max_error = max(max_error, float(np.max(np.abs(back-forward))))
    del weights
    c.base.atomic_npz(c.ROOT/'exact_maps.npz', totals=exact,
                     lattice_counts=np.array([m.sum() for m in lattice_masks]))
    c.save(c.ROOT/'geometry_ready.json', dict(complete=True, manifest_sha256=sha,
        sample_sha256=ready['sample_sha256'], weights_sha256=c.digest(c.ROOT/'adjoint_weights.npy'),
        exact_maps_sha256=c.digest(c.ROOT/'exact_maps.npz'), operators=c.OPERATORS, regions=c.REGIONS,
        actual_forward_particle_checks=len(picks), max_forward_absolute_error=max_error,
        normalized_models=list(c.MODELS[:2]), zero_padded_models=list(c.MODELS[2:])))


if __name__ == '__main__':
    main()
