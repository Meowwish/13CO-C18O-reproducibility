"""Audit paired controls without interpreting a stratified pilot as a galaxy map."""
import csv
import json
import os
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
CFG = json.loads(Path(os.environ.get('CONTROL_CONFIG', BASE / 'config.json')).read_text())
ROOT = BASE / CFG.get('output_subdir', 'output')
BRANCHES = ('actual_on', 'actual_off', 'uniform_on', 'uniform_off')


def audit(number, suffix):
    source = ROOT / f'snapshot{number}_paired{suffix}.h5'
    with h5py.File(source) as f:
        assert bool(f.attrs['complete'])
        names = json.loads(f.attrs['output_names_json'])
        inputs = f['actual_inputs'][:]
        radius = f['radius_kpc'][:]
        mass = f['mass_Msun'][:]
        probability = f['inclusion_probability'][:]
        strata = f['stratum'][:]
        status = {b: f[b + '/status'][:] for b in BRANCHES}
        values = {b: dict(zip(names, f[b + '/values'][:].T)) for b in BRANCHES}
    assert np.all((probability > 0) & (probability <= 1))
    weight = mass / probability
    good = np.logical_and.reduce([status[b] == 1 for b in BRANCHES])
    result = dict(snapshot=number, input=str(source), common_converged=int(good.sum()),
                  selected=len(good), mass_weighted_sample_missing_fraction=float(mass[~good].sum()/mass.sum()),
                  inverse_probability_mass_missing_fraction=float(weight[~good].sum()/weight.sum()),
                  branch_failures={b:int(np.sum(status[b] != 1)) for b in BRANCHES},
                  warning='Inverse-probability estimates from a small stratified pilot; not validated galaxy totals or x8 measurements.')
    comparisons = []
    rng = np.random.default_rng(20261009 + number)
    _, groups = np.unique(strata, axis=0, return_inverse=True)
    for numerator, denominator in [('actual_on','actual_off'), ('uniform_on','uniform_off'),
                                    ('actual_on','uniform_on'), ('actual_off','uniform_off')]:
        a, b = values[numerator], values[denominator]
        mask = good.copy()
        for v in (a,b):
            for k in ('L_13CO_1_0', 'L_C18O_1_0', 'X_13CO', 'X_C18O'):
                mask &= np.isfinite(v[k]) & (v[k] > 0)
        if not mask.any():
            continue
        line = np.log10(a['L_13CO_1_0'][mask]/a['L_C18O_1_0'][mask]) - np.log10(b['L_13CO_1_0'][mask]/b['L_C18O_1_0'][mask])
        molecular = np.log10(a['X_13CO'][mask]/a['X_C18O'][mask]) - np.log10(b['X_13CO'][mask]/b['X_C18O'][mask])
        strong = np.abs(line) > .1
        record = dict(numerator=numerator, denominator=denominator, positive_pairs=int(mask.sum()),
            line_ratio_delta_percentiles_dex=np.percentile(line,[0,5,50,95,100]).tolist(),
            molecular_ratio_delta_percentiles_dex=np.percentile(molecular,[0,5,50,95,100]).tolist(),
            sample_fraction_abs_line_delta_gt_0p1=float(strong.mean()))
        totals = {}
        for branch, v in ((numerator,a),(denominator,b)):
            totals[branch] = {}
            for k in ('L_CO_1_0','L_13CO_1_0','L_C18O_1_0'):
                w = weight[mask]*v[k][mask]
                total = w.sum()
                totals[branch][k] = float(total)
                record[branch+'_'+k+'_strong_response_fraction'] = float(w[strong].sum()/total) if total > 0 else None
        la, lb = totals[numerator], totals[denominator]
        record['pilot_weighted_sum_ratio_delta_dex'] = float(np.log10(
            (la['L_13CO_1_0']/la['L_C18O_1_0'])/(lb['L_13CO_1_0']/lb['L_C18O_1_0'])))
        for branch, v in ((numerator,a),(denominator,b)):
            for k in ('L_13CO_1_0','L_C18O_1_0'):
                w = weight[mask]*v[k][mask]
                record[branch+'_'+k+'_effective_sample_size'] = float(w.sum()**2/np.sum(w*w))
                record[branch+'_'+k+'_max_weight_fraction'] = float(w.max()/w.sum())
        # Paired resampling retains all four solutions of each row together.
        # It is conditional on successful positive rows, not a correction for failures.
        pools = [np.flatnonzero(mask & (groups == g)) for g in np.unique(groups)]
        pools = [p for p in pools if len(p)]
        draws = []
        for _ in range(1000):
            idx = np.concatenate([rng.choice(p, len(p), replace=True) for p in pools])
            ar = np.sum(weight[idx]*a['L_13CO_1_0'][idx])/np.sum(weight[idx]*a['L_C18O_1_0'][idx])
            br = np.sum(weight[idx]*b['L_13CO_1_0'][idx])/np.sum(weight[idx]*b['L_C18O_1_0'][idx])
            draws.append(np.log10(ar/br))
        record['conditional_stratified_bootstrap_95_percent_dex'] = np.percentile(draws,[2.5,97.5]).tolist()
        record['bootstrap_caveat'] = 'Conditional on observed successful pairs; excludes missing-solution, finite-population, model and spatial-covariance corrections. Singleton strata have no resampling variability.'
        comparisons.append(record)
    result['comparisons'] = comparisons
    rows = []
    fig, axes = plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for axrow, mixture in zip(axes, ('actual','uniform')):
        a, b = values[mixture+'_on'], values[mixture+'_off']
        mask = good & (a['L_13CO_1_0'] > 0) & (a['L_C18O_1_0'] > 0) & (b['L_13CO_1_0'] > 0) & (b['L_C18O_1_0'] > 0)
        delta = np.full(len(mask),np.nan)
        delta[mask] = np.log10(a['L_13CO_1_0'][mask]/a['L_C18O_1_0'][mask]) - np.log10(b['L_13CO_1_0'][mask]/b['L_C18O_1_0'][mask])
        for ax,x,label in zip(axrow,(a['Tg_K'], inputs[:,0],a['tau_13CO_1_0']),
                              ('Exchange-on temperature [K]', 'nH [cm^-3]', 'Exchange-on 13CO optical depth')):
            use = mask & (x > 0)
            ax.scatter(x[use],delta[use],s=9,alpha=.6)
            ax.set(xscale='log',xlabel=label,ylabel='Line-ratio change [dex]',title=mixture)
            ax.axhline(0,color='.4',ls=':')
        for lo,hi in ((0,30),(30,100),(100,300),(300,float('inf'))):
            use = mask & (a['Tg_K'] >= lo) & (a['Tg_K'] < hi)
            rows.append(dict(mixture=mixture,T_lo=lo,T_hi=hi,count=int(use.sum()),
                median_line_delta_dex=float(np.median(delta[use])) if use.any() else None))
    fig.suptitle(f'Original snapshot {number}: paired exchange response, stratified sample')
    for ext in ('png','pdf'):
        fig.savefig(ROOT/f'snapshot{number}_mechanism_audit{suffix}.{ext}',dpi=180)
    plt.close(fig)
    with (ROOT/f'snapshot{number}_temperature_bins{suffix}.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (ROOT/f'snapshot{number}_audit{suffix}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    audit(int(os.environ['CONTROL_SNAPSHOT']), '_retry' if os.environ.get('DIRECT_RUN_MODE') == 'retry' else '')
