"""Small design and edge-operator tests; never run DESPOTIC on login."""
import itertools
import json

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

import outer_common as c
from prepare_geometry import adjoint
from sfr_geometry import splat
from review import ratio_measure


def main():
    # Allocation must protect each outer annulus, preserve nonzero inclusion
    # across the full frame and census tiny occupied strata.
    population = np.full(480, 10000, 'i8')
    population[450:] = 0
    population[450:453] = [1, 2, 4]
    allocation = c.allocate(population)
    np.testing.assert_array_equal(allocation.reshape(16, -1).sum(1),
                                  np.r_[c.RADIAL_TARGETS[:15], 7])
    assert np.all(allocation <= population)
    assert np.all(allocation[:450] >= 2)
    empty = c.allocate(np.zeros(480, 'i8'))
    assert empty.sum() == 0
    # Exhaustive two-stratum SRSWOR enumeration checks inverse-probability
    # totals and the finite-population variance, including cross-covariance.
    y = np.array([[1., 3.], [2., 1.], [5., 7.], [9., 2.], [2., 8.], [6., 2.], [8., 4.]])
    totals, variances = [], []
    for a, b in itertools.product(itertools.combinations(range(4), 2),
                                  itertools.combinations(range(4, 7), 2)):
        indices = list(a)+list(b)
        scores = y[indices]/np.array([.5, .5, 2/3, 2/3])[:, None]
        totals.append(scores.sum(0))
        variances.append(c.covariance(scores, np.array([0, 0, 1, 1]), np.array([4, 3]), np.array([2, 2])))
    np.testing.assert_allclose(np.mean(totals, axis=0), y.sum(0), rtol=1e-13)
    np.testing.assert_allclose(np.mean(variances, axis=0), np.cov(np.array(totals).T, bias=True), rtol=1e-13)
    # Nearest-edge bilinear splatting must be the exact transpose of sampling,
    # for both normalized and zero-padded Gaussian operators.
    rng = np.random.default_rng(575)
    field = rng.random((19, 23))
    coords = np.array([[-.4, 3.2], [18.6, 22.7], [3.8, 4.2], [18., 0.], [0., 22.]])
    mask = np.array([1., 2., 0., 3., 4.])
    raw_splat = splat(mask, coords, field.shape)
    errors = []
    for normalized in (False, True):
        smoothed = gaussian_filter(field, 1.3, mode='constant', cval=0., truncate=4.)
        if normalized:
            smoothed /= gaussian_filter(np.ones_like(field), 1.3, mode='constant', cval=0., truncate=4.)
        forward = mask@map_coordinates(smoothed, coords.T, order=1, mode='nearest', prefilter=False)
        backward = np.sum(field*adjoint(raw_splat, 1.3, normalized))
        np.testing.assert_allclose(backward, forward, rtol=2e-13, atol=2e-13)
        errors.append(float(abs(backward-forward)))
    # Numerical nonresponse remains excluded, with a reported proxy-light
    # fraction; it must not be filled with surrogate values.
    direct = np.tile([2., 1.], (4, 1))
    proxy = direct.copy()
    qualified = np.array([True, True, False, True])
    direct[2] = np.nan
    measured = ratio_measure(direct, proxy, qualified, np.ones((4, len(c.REGIONS))),
                             np.ones(4), np.zeros(4, int), np.array([4]), np.array([4]))
    for r in measured['rows']:
        np.testing.assert_allclose(r['direct_minus_surrogate_dex']['point'], 0., atol=1e-14)
        np.testing.assert_allclose(r['missing_predicted_positive_line_fractions'], [.25, .25])
    c.save(c.ROOT/'local_checks.json', dict(passed=True, physical_solves=0,
        allocation_checked=True, exhaustive_SRSWOR_checked=True, nonresponse_checked=True,
        bilinear_gaussian_adjoint_absolute_errors=errors))
    print('Local design, covariance, nonresponse and edge-adjoint tests passed.')


if __name__ == '__main__':
    main()
