"""Sparse evaluation of the production Meshoid + beam + sampling operator.

The native deposition grid uses size/(res-1), while the published sampling
coordinate uses size/res. Both conventions are deliberately reproduced.
"""
import numpy as np
from numba import njit
from meshoid.kernel_density import kernel2d
from scipy.ndimage import gaussian_filter


def make_patches(xy):
    from sfr_geometry import splat
    shape = (1500, 1500)
    sigma = 125/(np.sqrt(8*np.log(2))*20)
    den = gaussian_filter(np.ones(shape), sigma, mode='constant', cval=0., truncate=4.)
    patches = np.zeros((len(xy), 2, 26, 26)); bounds = np.zeros((len(xy), 4), 'i8')
    reach = 0.
    for a, point in enumerate(xy):
        s = splat([1.], ((point[None, :]+15)/.02-.5), shape)
        fields = [gaussian_filter(s/den if op == 0 else s, sigma, mode='constant', cval=0., truncate=4.) for op in range(2)]
        i, j = np.nonzero(fields[0]); x0, x1, y0, y1 = i.min(), i.max()+1, j.min(), j.max()+1
        assert x1-x0 <= 26 and y1-y0 <= 26
        bounds[a] = x0, x1, y0, y1
        for op in range(2):
            patches[a, op, :x1-x0, :y1-y0] = fields[op][x0:x1, y0:y1]
        corners = np.array([[x, y] for x in (x0, x1-1) for y in (y0, y1-1)])*30000/1499-15000
        reach = max(reach, float(np.max(np.linalg.norm(corners-point*1000, axis=1))))
    return patches, bounds, reach


@njit
def evaluate(xyz, smoothing, particle, aperture, patches, bounds):
    result = np.zeros((len(particle), 2))
    dx = 30000./1499
    for pair in range(len(particle)):
        i, a = particle[pair], aperture[pair]
        # Preserve the production pre-filter using the original smoothing h.
        if abs(xyz[i, 0]) > 15000+smoothing[i] or abs(xyz[i, 1]) > 15000+smoothing[i]:
            continue
        h = max(40., smoothing[i]); hinv = 1./h
        x, y = xyz[i, 0]+15000, xyz[i, 1]+15000
        bx0, bx1, by0, by1 = bounds[a]
        x0 = max(int((x-h)/dx+1), 0, bx0); x1 = min(int((x+h)/dx), 1499, bx1-1)
        y0 = max(int((y-h)/dx+1), 0, by0); y1 = min(int((y+h)/dx), 1499, by1-1)
        for gx in range(x0, x1+1):
            for gy in range(y0, y1+1):
                r = np.sqrt((x-gx*dx)**2+(y-gy*dx)**2)
                if r > h:
                    continue
                deposit = kernel2d(r*hinv)*hinv*hinv
                for op in range(2):
                    result[pair, op] += patches[a, op, gx-bx0, gy-by0]*deposit
    return result
