"""Small geometry, selection-boundary and spawned-worker import checks."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import tempfile

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter

import aperture_common as c
from kernel import make_patches, evaluate
from aperture_statistics import detection, median, summaries


def spawned_import():
    import solve
    import review
    import prepare
    assert callable(solve.worker.initialize) and callable(solve.worker.solve)
    return True


def main():
    xy = np.array([[0., 0.], [14.999, .02], [-14.999, -14.97], [10.12, -5.17]])
    patches, bounds, reach = make_patches(xy)
    xyz = np.array([[0., 0., 10.], [14990, 30, -200], [-15030, -14960, 0.], [10150., -5190., 0.]])
    h = np.array([.8, 17., 65., 250.])
    pairs = np.array([(i, a) for i in range(4) for a in range(4)])
    result = evaluate(xyz, h, pairs[:, 0], pairs[:, 1], patches, bounds)
    from sfr_geometry import project
    from plot_galic_selection import sample
    sigma = 125/(np.sqrt(8*np.log(2))*20)
    den = gaussian_filter(np.ones((1500, 1500)), sigma, mode='constant', cval=0., truncate=4.)
    maximum = 0.
    for i in range(4):
        native = project(xyz[i:i+1], h[i:i+1])
        beam = gaussian_filter(native, sigma, mode='constant', cval=0., truncate=4.)
        for op in range(2):
            expected = sample(beam/den if op == 0 else beam, xy)
            got = result[i*4:(i+1)*4, op]
            np.testing.assert_allclose(got, expected, rtol=5e-12, atol=1e-22)
            maximum = max(maximum, float(np.max(np.abs(got-expected))))
    # The KD-tree geometric bound must include every nonzero pair.
    for (i, a), value in zip(pairs, result):
        if np.any(value > 0):
            assert np.linalg.norm(xyz[i, :2]-xy[a]*1000) <= max(h[i], 40)+reach
    line = np.array([[.489, .432], [.49, .433], [.6, .433], [.1, .1]])
    np.testing.assert_array_equal(detection(line, 1), [False, True, True, False])
    assert not detection(line, 1, 'dual_and_ratio_snr').any()
    assert median([1, 2, 3], [1, 8, 1]) == 2
    one = summaries(np.array([[4., 1.], [12., 3.]]), np.array([2., 1.]), np.ones(2, bool))
    np.testing.assert_allclose(one, [4, 4, 4, 1])
    from solve import read_rows
    with tempfile.TemporaryDirectory() as td:
        with h5py.File(td+'/rows.h5', 'w') as f:
            a = np.arange(200005*2).reshape(-1, 2); d = f.create_dataset('a', data=a)
            indices = np.array([1, 5, 99999, 100000, 200004])
            np.testing.assert_array_equal(read_rows(d, indices), a[indices])
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
        assert pool.submit(spawned_import).result(timeout=90)
    c.save(c.ROOT/'local_checks.json', dict(passed=True, actual_forward_checks=8,
        maximum_forward_absolute_difference=maximum, candidate_bound_checked=True,
        threshold_equality_checked=True, weighted_statistics_checked=True,
        chunked_input_reads_checked=True, spawned_imports_checked=True, chemical_solves_on_login=0))
    print('Aperture local checks passed.', flush=True)


if __name__ == '__main__':
    main()
