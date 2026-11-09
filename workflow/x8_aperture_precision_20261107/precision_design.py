"""Residual-total estimator, exact SRS variance, and selection envelopes."""
import numpy as np


def allocate(population, importance, count):
    population = np.asarray(population, dtype='i8')
    count = min(int(count), int(population.sum()))
    n = np.minimum(population, 2)
    assert n.sum() <= count
    while n.sum() < count:
        available = n < population
        priority = np.where(available, importance/(n+1), -np.inf)
        if np.max(priority) <= 0:
            priority = np.where(available, population-n, -np.inf)
        n[int(np.argmax(priority))] += 1
    return n


def total_and_covariance(baseline, residual, pi, strata, population):
    score = np.asarray(residual)/pi[:, None]
    total = np.asarray(baseline)+score.sum(axis=0)
    covariance = np.zeros((score.shape[1], score.shape[1]))
    for h in np.unique(strata):
        z = score[strata == h]; nh = len(z); N = int(population[h])
        assert 0 < nh <= N
        if nh < N:
            assert nh >= 2
            covariance += nh*(1-nh/N)*np.cov(z, rowvar=False, ddof=1)
    return total, covariance


def median_bounds(low, high, weight, mandatory, possible):
    """Enclose the weighted median for every compatible selected catalogue.

    Inclusion of mandatory objects is fixed; optional objects may enter or
    leave. Their values can take any position in their supplied intervals.
    """
    optional = possible&(~mandatory)
    M = weight[mandatory].sum()
    if M <= 0:
        return np.array([np.nan, np.nan])
    grid = np.unique(np.r_[low[possible], high[possible]])
    below_lo = low[None, :] <= grid[:, None]
    below_hi = high[None, :] <= grid[:, None]
    optional_low = below_lo@(weight*optional)
    fmax = (below_lo@(weight*mandatory)+optional_low)/(M+optional_low)
    fmin = (below_hi@(weight*mandatory))/(M+(~below_hi)@(weight*optional))
    return np.array([grid[np.flatnonzero(fmax >= .5)[0]], grid[np.flatnonzero(fmin >= .5)[0]]])


def selection(low, high, depth, mode):
    from aperture_statistics import detection
    mandatory = detection(low, depth, mode)
    possible = detection(high, depth, mode)
    assert np.all((~mandatory)|possible)
    return mandatory, possible


def selected_bounds(low, high, weight, depth, mode):
    mandatory, possible = selection(low, high, depth, mode)
    return catalogue_bounds(low, high, weight, depth, mandatory, possible)


def catalogue_bounds(low, high, weight, depth, mandatory, possible):
    result = np.full((4, 2), np.nan)
    result[3] = [weight[mandatory].sum()/weight.sum(), weight[possible].sum()/weight.sum()]
    if not mandatory.any():
        return result
    # Conditional on selection, both intensities exceed the line thresholds.
    floor = np.array([.489, .432])/depth
    lo = np.maximum(low, floor)
    hi = np.maximum(high, lo)
    ratio = median_bounds(lo[:, 0]/hi[:, 1], hi[:, 0]/lo[:, 1], weight, mandatory, possible)
    a = median_bounds(lo[:, 0], hi[:, 0], weight, mandatory, possible)
    b = median_bounds(lo[:, 1], hi[:, 1], weight, mandatory, possible)
    result[0] = ratio
    result[1] = [a[0]/b[1], a[1]/b[0]]
    lower = (weight[:, None]*mandatory[:, None]*lo).sum(axis=0)
    upper = (weight[:, None]*possible[:, None]*hi).sum(axis=0)
    result[2] = [lower[0]/upper[1], upper[0]/lower[1]]
    return result


def sum_ratio_bounds(low, high, weight, mandatory, possible):
    if not mandatory.any(): return np.array([np.nan, np.nan])
    lower = (weight[:, None]*mandatory[:, None]*np.maximum(low, 0.)).sum(axis=0)
    upper = (weight[:, None]*possible[:, None]*np.maximum(high, 0.)).sum(axis=0)
    if upper[1] <= 0: return np.array([np.nan, np.nan])
    return np.array([lower[0]/upper[1], upper[0]/lower[1] if lower[1] > 0 else np.inf])


def difference_bounds(direct, proxy):
    result = np.full((4, 2), np.nan)
    result[:3, 0] = np.log10(direct[:3, 0]/proxy[:3, 1])
    result[:3, 1] = np.log10(direct[:3, 1]/proxy[:3, 0])
    result[3] = [direct[3, 0]-proxy[3, 1], direct[3, 1]-proxy[3, 0]]
    return result
