"""Design-weighted aperture summaries; selection is recomputed for each map."""
import numpy as np


def detection(line, depth, mode='dual'):
    noise = np.array([.163, .144])/depth
    mask = np.all(line > 3*noise, axis=1)
    if mode == 'dual_and_ratio_snr':
        mask &= np.sum((noise/np.maximum(line, 1e-300))**2, axis=1) < 1/9
    return mask


def median(values, weight):
    values, weight = np.asarray(values), np.asarray(weight)
    keep = np.isfinite(values)&(weight > 0)
    if not keep.any():
        return np.nan
    order = np.argsort(values[keep], kind='stable')
    x, w = values[keep][order], weight[keep][order]
    # Inverse of the design-weighted empirical CDF, including the lower tie.
    return x[np.searchsorted(np.cumsum(w), w.sum()/2, side='left')]


def summaries(line, weight, mask):
    w = np.where(mask, weight, 0.)
    if not np.any(w > 0):
        return np.full(4, np.nan)
    total = np.sum(w[:, None]*line, axis=0)
    return np.array([median(line[:, 0]/np.maximum(line[:, 1], 1e-300), w),
        median(line[:, 0], w)/median(line[:, 1], w), total[0]/total[1], w.sum()/weight.sum()])
