"""Finite/analytic checks; no chemistry or full-frame inference."""
import ast
import itertools
import json
import subprocess

import numpy as np

import closure_common as c
from lvg_envelope import bounds, lamda_transition, PLANCK, LIGHT, BOLTZMANN, PC


def main():
    from collections import Counter
    frequencies = Counter()
    for ordering in itertools.permutations(range(6)):
        keys = np.argsort(ordering)
        local = np.r_[np.argsort(keys[:3])[:2], 3+np.argsort(keys[3:])[:2]]
        chosen = local[np.argsort(keys[local])[:2]]
        np.testing.assert_array_equal(np.sort(chosen), np.sort(np.argsort(keys)[:2]))
        frequencies[tuple(sorted(chosen))] += 1
    assert len(frequencies) == 15 and len(set(frequencies.values())) == 1
    rng = np.random.default_rng(20261109)
    raw = np.column_stack((10**rng.uniform(-1, 7, 20000), np.full(20000, 1e22),
        10**rng.uniform(-3, 2, 20000), np.full(20000, 2e-4),
        np.full(20000, 2e-6), np.full(20000, 4e-4), np.full(20000, 5e-7)))
    for iso, name in [('13CO', '13co'), ('C18O', 'c18o')]:
        t = lamda_transition(c.ISO/f'chemTable_v4_full_direct/LAMDA/{name}.dat')
        A, nu, g = t
        cap = np.minimum(raw[:, 4], raw[:, 5]) if iso == '13CO' else np.minimum(raw[:, 3], raw[:, 6])
        x = cap*rng.random(len(raw)); f = rng.dirichlet([1, 1, 1], len(raw)); f0, f1 = f[:, 0], f[:, 1]
        tau = A*(LIGHT/nu)**3*raw[:, 0]/(8*np.pi*raw[:, 2]*1e5/PC)*x*(g*f0-f1)
        beta = 1-tau/2
        use = tau > 1e-6; beta[use] = -np.expm1(-tau[use])/tau[use]
        photon = 1/np.expm1(PLANCK*nu/(BOLTZMANN*2.73))
        line = ((1+photon)*f1-g*photon*f0)*A*PLANCK*nu*x*beta
        box = bounds(raw, t, iso)
        assert np.all(line >= box[:, 0]) and np.all(line <= box[:, 1])
    # Exhaustive box extrema of a signed radial/SFR contrast.
    a = np.array([-.5, .2, .3]); box = np.array([[1., 2.], [3., 5.], [4., 6.]])
    low = sum(a[j]*box[j, int(a[j] < 0)] for j in range(3))
    high = sum(a[j]*box[j, int(a[j] >= 0)] for j in range(3))
    corners = [a@np.array([box[j, v] for j, v in enumerate(choice)]) for choice in itertools.product((0, 1), repeat=3)]
    assert np.isclose(low, min(corners)) and np.isclose(high, max(corners))
    # Exact SRS covariance at the census and finite-population limit.
    assert c.srs_variance(np.array([1., 4.]), np.array([0, 0]), np.array([2])) == 0
    assert np.isclose(c.srs_variance(np.array([1., 4.]), np.array([0, 0]), np.array([4])), 4.5)
    # Parse every new source and import the worker wrappers without initialising
    # DESPOTIC. The established worker has separately tested spawn semantics.
    for path in c.ROOT.glob('*.py'):ast.parse(path.read_text(), filename=str(path))
    for name in ('direct', 'temporal_prepare', 'temporal_geometry', 'review_temporal', 'selection_direct', 'review_recovery'):
        __import__(name)
    subprocess.run(['bash', '-n', str(c.ROOT/'run.pbs')], check=True)
    c.save(c.ROOT/'local_checks.json', dict(passed=True, priority_reservoir_permutations=720,
        equally_likely_srs_subsets=15, analytic_lvg_draws=40000, contrast_corners=8,
        covariance_checks=2, imports_without_chemistry=True, physical_solver_called=False))
    print('Finite sampling / LVG envelope / covariance / imports / syntax checks passed.')


if __name__ == '__main__':main()
