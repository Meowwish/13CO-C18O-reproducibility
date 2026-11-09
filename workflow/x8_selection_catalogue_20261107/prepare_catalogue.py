"""A separate full-frame probability cohort targeted at D1000 selection."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
ISO = ROOT.parent
sys.path.extend([str(ISO), str(ISO/'highres_snapshot575_temporal')])
from plot_galic_selection import lattice, sample

MODELS = ('production_full', 'production_slug', 'actual_on', 'actual_off', 'uniform_on', 'uniform_off')
TARGETS = np.array([32, 24, 16, 24, 8, 8])


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    assert int(os.environ.get('PBS_NCPUS', '0')) > 0
    assert not (ROOT/'catalogue.npz').exists()
    xy = lattice(); radius = np.hypot(*xy.T); fields = []
    paths = [ISO/'highres_snapshot575_temporal/output/snapshot_575_maps_125pc.npz',
        ISO/'highres_snapshot575_temporal/slug_only/output/snapshot_575_maps_125pc.npz',
        ISO/'isotope_fractionation_controls/x8_snapshot575/maps_125pc.npz']
    original = json.loads((ISO/'x8_outer_validation_20261105/manifest.json').read_text())
    for path in paths: assert sha(path) == original['hashes'][str(path)]
    for m, name in enumerate(MODELS):
        with np.load(paths[m] if m < 2 else paths[2]) as f:
            prefix = '' if m < 2 else name+'_'
            keys = ('W13CO', 'WC18O', 'C13O_abundance', 'C18O_abundance') if m < 2 else ('W13CO', 'WC18O', 'X13CO', 'XC18O')
            fields.append(np.column_stack([sample(f[prefix+k], xy) for k in keys]))
    fields = np.stack(fields, axis=1)
    detected = np.all(fields[:, :, :2] > np.array([3*.000163, 3*.000144]), axis=2)
    common = detected[:, 2:6].all(axis=1)
    sources = detected[:, :2].all(axis=1)
    any_detection = detected.any(axis=1)
    margin = np.max(np.min(fields[:, :, :2]/np.array([3*.000163, 3*.000144]), axis=2), axis=1)
    category = np.select([common, sources, any_detection, margin >= .7, margin >= .25], [0, 1, 2, 3, 4], default=5)
    rng = np.random.default_rng(202611075751000)
    selected = []; populations = []; allocations = []; frame_stratum = np.full(len(xy), -1, 'i8')
    for radial in range(8, 15):
        for group, target in enumerate(TARGETS):
            members = np.flatnonzero((radius >= radial)&(radius < radial+1)&(category == group))
            h = len(populations); nh = min(len(members), int(target)); frame_stratum[members] = h
            populations.append(len(members)); allocations.append(nh)
            if nh: selected.extend(rng.choice(members, nh, replace=False).tolist())
    selected = np.array(sorted(selected)); population, allocation = np.array(populations), np.array(allocations)
    assert len(np.unique(selected)) == len(selected) <= 7*TARGETS.sum()
    assert np.all((population == 0)|(allocation > 0))
    assert population.sum() == np.sum((radius >= 8)&(radius < 15))
    strata = frame_stratum[selected]; pi = allocation[strata]/population[strata]
    np.testing.assert_array_equal(np.bincount(strata, minlength=len(population)), allocation)
    np.savez_compressed(ROOT/'catalogue.npz', xy=xy[selected], lattice_index=selected, stratum=strata,
        inclusion_probability=pi, population=population, allocation=allocation, predicted=fields[selected],
        detection_flags=detected[selected], seed=np.asarray(202611075751000))
    counts = []
    for r in range(8, 15):
        mask = (radius[selected] >= r)&(radius[selected] < r+1)
        d = detected[selected][mask]
        counts.append(dict(annulus_kpc=[r, r+1], drawn=int(mask.sum()),
            expected_full_detections=int(d[:, 0].sum()), expected_slug_detections=int(d[:, 1].sum()),
            expected_common_control_detections=int(d[:, 2:6].all(axis=1).sum())))
    report = dict(complete=True, probability_frame='All published 62.5 pc lattice positions at 8<=R<15 kpc',
        distinct_apertures=len(selected), maximum_apertures=int(7*TARGETS.sum()),
        stratum_population=population.tolist(), stratum_allocation=allocation.tolist(), expected_detection_counts=counts,
        all_nonempty_strata_have_positive_inclusion=True, model_names=MODELS,
        input_hashes={str(path): sha(path) for path in paths}, catalogue_sha256=sha(ROOT/'catalogue.npz'),
        chemistry_submitted=False, labels_consulted=False,
        scope='Independent D1000-targeted probability cohort; do not pool with the earlier broad cohort. '
            'Particle sampling intensity will be set after the running precision pilot is reviewed.')
    (ROOT/'prepared.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Prepared D1000 cohort:', len(selected), 'apertures;', counts, flush=True)


if __name__ == '__main__': main()
