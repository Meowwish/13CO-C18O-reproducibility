"""Direct checks at three new times using matched projected operators."""
import json
import os

import h5py
import numpy as np

import closure_common as c


def main():
    assert int(os.environ.get('PBS_NCPUS', '0')) > 0
    manifest = c.verify(); snap = int(os.environ['SNAPSHOT']); assert snap in c.SNAPSHOTS
    folder = c.ROOT/f'temporal{snap}'; ready = json.loads((folder/'prepared.json').read_text())
    values, qualified, status = c.assemble(f'temporal{snap}')
    n = ready['particles']; values = values.reshape(n, 5, 27); qualified = qualified.reshape(n, 5)
    with h5py.File(folder/'sample.h5') as f:
        strata, pop, pi, nuclei = (f[k][:] for k in ('stratum', 'population', 'inclusion_probability', 'nuclei'))
        prediction = f['prediction'][:]
    with np.load(folder/'geometry.npz') as f:
        assert bool(f['complete']) and str(f['sample_sha256']) == ready['sample_sha256']
        kernel = f['kernel'][:]
    offset = 3*np.log10(109782173400/110201354300)
    checks = []; pairs = []; shapes = []
    region_names = [f'annulus_{r}_{r+1}' for r in range(15)] + ['inner_1_8', 'outer_8_15', 'disc_0_15', 'full_field']
    common_controls = qualified[:, [0, 2, 3, 4]].all(axis=1)
    def measure(model, mask, region, geometry):
        operator = geometry*2+int(model >= 2)
        weight = nuclei/pi*kernel[:, operator, region]
        direct = c.estimate(values[:, c.MAP_BRANCH[model], 7:9], weight, mask, strata, pop)
        proxy = c.estimate(prediction[:, model, 4:6], weight, mask, strata, pop)
        return direct, proxy
    for geometry, geometry_name in enumerate(('native', 'lattice')):
        for model, name in enumerate(c.MODELS):
            q = qualified[:, c.MAP_BRANCH[model]] if model < 2 else common_controls
            radial_points = []; radial_scores = []
            for r, region in enumerate(region_names):
                d, p = measure(model, q, r, geometry)
                if d is None or p is None:
                    checks.append(dict(geometry=geometry_name, model=name, region=region, finite=False))
                    if 8 <= r < 15:radial_points.append(np.nan); radial_scores.append(np.zeros(n))
                    continue
                dr = c.result(d[0]+offset, d[1], strata, pop)
                er = c.result(d[0]-p[0], d[1]-p[1], strata, pop)
                ci = er['interval_95']
                checks.append(dict(geometry=geometry_name, model=name, region=region, finite=True,
                    direct_log_brightness_ratio=dr, matched_surrogate_log_brightness_ratio=float(p[0]+offset),
                    direct_minus_surrogate_dex=er, interval_within_0p05_dex=ci[0] >= -.05 and ci[1] <= .05))
                if 8 <= r < 15:radial_points.append(d[0]); radial_scores.append(d[1])
            if np.isfinite(radial_points).all():
                x = np.arange(8.5, 15.); a = (x-x.mean())/np.sum((x-x.mean())**2)
                slope = c.result(a@radial_points, np.stack(radial_scores, axis=1)@a, strata, pop)
                shapes.append(dict(geometry=geometry_name, model=name, contrast='outer_slope_dex_per_kpc', direct=slope))
        for label, one, two in [('source', 1, 0), ('exchange', 2, 3), ('uniform_exchange', 4, 5)]:
            q = qualified[:, c.MAP_BRANCH[[one, two]]].all(axis=1)
            for r in (15, 16, 18):
                d1, p1 = measure(one, q, r, geometry); d2, p2 = measure(two, q, r, geometry)
                if any(x is None for x in (d1, p1, d2, p2)):continue
                direct = c.result(d1[0]-d2[0], d1[1]-d2[1], strata, pop)
                error = c.result((d1[0]-d2[0])-(p1[0]-p2[0]), (d1[1]-d2[1])-(p1[1]-p2[1]), strata, pop)
                pairs.append(dict(geometry=geometry_name, comparison=label, region=region_names[r],
                    direct_contrast=direct, contrast_error=error))
    c.save(folder/'review.json', dict(complete=True, snapshot=snap, manifest_sha256=manifest,
        particles=n, qualified_states=int(qualified.sum()), failed=int((status == -1).sum()),
        timeout=int((status == -2).sum()), checks=checks, paired_checks=pairs, shape_checks=shapes,
        model_fitting=False, scientific_acceptance=False,
        scope='Independent probability sample at a new time; native/lattice exact adjoints; direct and proxy share qualified states. Pointwise sampling intervals exclude physical-model and unknown nonresponse light.'))
    print('New-time direct reference complete:', snap, flush=True)


if __name__ == '__main__':main()
