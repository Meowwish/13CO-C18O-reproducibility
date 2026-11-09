"""Rebuild small result tables/plots from published machine-readable results."""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parent


def verify():
    manifest = json.loads((ROOT/'FILE_SHA256.json').read_text())
    for name, expected in manifest.items():
        path = ROOT/name
        assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    return len(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reproduced')
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args(); count = verify()
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    d = json.loads((ROOT/'results/global_integrals.json').read_text())
    total = np.asarray(d['complete_frame_line_integrals'])
    freq = d['frequencies_Hz']; ratio = total[:, 1]/total[:, 2]*(freq[2]/freq[1])**3
    expected = np.array([8.268493532885843, 6.488346623524451, 8.257177152020528,
                         6.492090793253663, 7.736421080038711, 5.994283669284608])
    np.testing.assert_allclose(ratio, expected, rtol=2e-12)
    responses = dict(source_removal_percent=100*(ratio[1]/ratio[0]-1),
                     exchange_actual_percent=100*(ratio[2]/ratio[3]-1),
                     exchange_uniform_percent=100*(ratio[4]/ratio[5]-1))
    with (out/'global_ratios.csv').open('w') as f:
        w = csv.writer(f);w.writerow(['model', 'brightness_ratio'])
        w.writerows(zip(d['model_order'], ratio))
    selected = json.loads((ROOT/'results/x8_selection_reference_20261108/review.json').read_text())
    statistics = ['ratio_of_line_sums', 'median_local_ratio', 'ratio_of_line_medians']
    precision_rows = []
    for model in d['model_order']:
        counts = []
        for stat in statistics:
            rows = [x for x in selected['selected_statistics'] if x['model'] == model and x['selection'] == 'dual'
                    and x['depth'] == 1000 and x['statistic'] == stat]
            assert len(rows) == 7 and all(x['sufficient_certain_detections'] for x in rows)
            passed = [x for x in rows if np.isfinite(x['reference_and_pixel_sampling_interval_95']).all()
                      and x['reference_and_pixel_sampling_interval_95'][0] >= -.05
                      and x['reference_and_pixel_sampling_interval_95'][1] <= .05]
            assert len(passed) == sum(x['interval_within_0p05_dex'] for x in rows)
            counts.append(len(passed))
        precision_rows.append([model, *counts])
    assert [x[1:] for x in precision_rows] == [[6, 3, 1], [3, 1, 1], [6, 2, 2], [6, 2, 2], [3, 3, 0], [4, 2, 0]]
    with (out/'selected_precision_table.csv').open('w') as f:
        w = csv.writer(f);w.writerow(['model', *statistics]);w.writerows(precision_rows)
    outer = json.loads((ROOT/'results/x8_outer_validation_20261105/review.json').read_text())
    offset = outer['ratio_offset_energy_to_brightness_dex']; profiles = []
    for model in d['model_order']:
        for r in range(8, 15):
            row = next(x for x in outer['model_checks'] if x['geometry'] == 'lattice' and x['model'] == model
                       and x['region'] == f'annulus_{r}_{r+1}')
            ref = row['direct_log_energy_ratio']; point = 10**(ref['point']+offset)
            lo, hi = 10**(np.asarray(ref['interval_95'])+offset)
            profiles.append([model, r, r+1, point, lo, hi])
    with (out/'direct_outer_profiles.csv').open('w') as f:
        w = csv.writer(f);w.writerow(['model','r_lo_kpc','r_hi_kpc','ratio','low95','high95']);w.writerows(profiles)
    # Export all structured result records, preserving nested intervals and
    # qualifications. This is a table export, not independent particle replay.
    exports = 0
    for path in sorted((ROOT/'results').rglob('*.json')):
        data = json.loads(path.read_text())
        if not isinstance(data, dict):continue
        for key, rows in data.items():
            if not isinstance(rows, list) or not rows or not all(isinstance(x, dict) for x in rows):continue
            target = out/'records'/path.relative_to(ROOT/'results').parent/(path.stem+'_'+key+'.csv')
            target.parent.mkdir(parents=True, exist_ok=True)
            fields = sorted({k for row in rows for k in row})
            with target.open('w') as f:
                w = csv.DictWriter(f, fieldnames=fields);w.writeheader()
                w.writerows({k:json.dumps(v) if isinstance(v, (list, dict)) else v for k,v in row.items()} for row in rows)
            exports += 1
    if args.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6,4), layout='constrained')
        for model in ['production_full', 'uniform_off']:
            rows = [x for x in profiles if x[0] == model]; x = np.array([r[1]+.5 for r in rows]);y = np.array([r[3] for r in rows])
            ax.errorbar(x, y, yerr=[y-np.array([r[4] for r in rows]),np.array([r[5] for r in rows])-y], fmt='o-', label=model)
        ax.set(xlabel='Radius [kpc]', ylabel='Direct brightness ratio');ax.legend()
        fig.savefig(out/'direct_outer_profiles.pdf');plt.close(fig)
    report = dict(passed=True, verified_public_files=count, global_ratios=ratio.tolist(), responses=responses,
                  exported_record_tables=exports, selected_precision_rows=precision_rows,
                  scope='Saved-integral reconstruction and machine-readable table export. Cached particle/covariance/model replay is a separate requested-data mode.')
    (out/'reproduction_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':main()
