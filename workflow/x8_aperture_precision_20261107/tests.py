"""Exact finite-population and catalogue-envelope tests; no login cloud solves."""
from concurrent.futures import ProcessPoolExecutor
import itertools
import multiprocessing

import numpy as np

import precision_common as c
from precision_design import allocate, total_and_covariance, median_bounds, selected_bounds, difference_bounds
from aperture_statistics import median


def spawned_imports():
    import precision_prepare as prepare
    import precision_solve as solve
    import precision_review as review
    assert str(c.ROOT) in prepare.__file__ and str(c.ROOT) in solve.__file__ and str(c.ROOT) in review.__file__
    assert callable(solve.worker.solve)
    return True


def main():
    # Exhaust all stratified samples. The difference total is unbiased, and
    # mean estimated covariance equals the actual randomization covariance.
    actual = np.array([[1., 2.], [4., 1.], [2., 7.], [9., 4.], [2., 4.], [6., 2.], [4., 3.]])
    proxy = np.array([[2., 2.], [3., 2.], [3., 4.], [7., 3.], [3., 3.], [5., 3.], [4., 3.]])
    baseline = proxy.sum(axis=0); population = np.array([4, 3])
    points = []; covariances = []
    for one in itertools.combinations(range(4), 2):
        for two in itertools.combinations(range(4, 7), 2):
            ids = np.array(one+two); strata = np.array([0, 0, 1, 1]); pi = np.array([.5, .5, 2/3, 2/3])
            point, cv = total_and_covariance(baseline, actual[ids]-proxy[ids], pi, strata, population)
            points.append(point); covariances.append(cv)
    np.testing.assert_allclose(np.mean(points, axis=0), actual.sum(axis=0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.mean(covariances, axis=0), np.cov(points, rowvar=False, ddof=0), rtol=0, atol=1e-12)
    n = allocate(np.array([1, 3, 20, 0]), np.array([0., 0., 5., 0.]), 12)
    assert n.sum() == 12 and np.all(n <= [1, 3, 20, 0]) and np.all(n[[0, 1, 2]] > 0)
    # Exhaust all optional memberships and interval endpoints. The enclosure
    # must cover every compatible weighted median, including unequal weights.
    low = np.array([1., 2., 1.5, 5., 3.]); high = np.array([3., 4., 6., 7., 8.])
    weight = np.array([1., 3., 2., 1., 2.]); mandatory = np.array([True, True, False, False, False]); possible = np.ones(5, bool)
    bounds = median_bounds(low, high, weight, mandatory, possible)
    checked = 0
    for chosen in itertools.product((False, True), repeat=3):
        mask = mandatory.copy(); mask[2:] = chosen
        for ends in itertools.product((False, True), repeat=5):
            values = np.where(ends, high, low); m = median(values[mask], weight[mask])
            assert bounds[0] <= m <= bounds[1]; checked += 1
    line = np.array([[4., 1.], [12., 3.], [.01, .001]])
    exact = selected_bounds(line, line, np.array([1., 2., 3.]), 1, 'dual')
    np.testing.assert_allclose(exact[:3], np.full((3, 2), 4.))
    np.testing.assert_allclose(difference_bounds(exact, exact), np.zeros((4, 2)), atol=1e-12)
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
        assert pool.submit(spawned_imports).result(timeout=90)
    c.save(c.ROOT/'local_checks.json', dict(passed=True, exhaustive_stratified_draws=len(points),
        unbiased_residual_total=True, randomization_covariance_matches=True,
        catalogue_configurations_enclosed=checked, exact_reference_identity=True,
        positive_tail_inclusion=True, spawned_imports=True, chemical_solves_on_login=0))
    print('Finite-population and catalogue-envelope tests passed.', flush=True)


if __name__ == '__main__':
    main()
