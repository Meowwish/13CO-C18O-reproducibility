"""Summarize audited central-claim checks without modifying the manuscript."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
LABELS=dict(actual_on='Actual isotopes; exchange on',actual_off='Actual isotopes; exchange off',
    uniform_on='Uniform isotopes; exchange on',uniform_off='Uniform isotopes; exchange off')


def interval(record):
    q=record['interval_95']
    return f'{record["point"]:+.4f} [{q[0]:+.4f}, {q[1]:+.4f}]' if q else 'undefined'


def main():
    audit=json.loads((ROOT/'independent_audit.json').read_text());assert audit['passed']
    d=json.loads((ROOT/'review.json').read_text())
    exposure=json.loads((ROOT/'shielding_exposure.json').read_text())
    rows=['# Central-claim checks: audited conditional references','',
        'This review uses existing direct labels, frozen predictions and matched probability-weighted samples.',
        'No training, new chemistry, full-population accuracy claim or automatic paper edit is performed.',
        'Intervals describe stratified sampling conditional on successful states, not physical uncertainty.','',
        f'Independent reconstruction passed for {audit["independently_checked_intervals"]} interval sets.','',
        '## Inner-annulus precision, extended references','',
        '| Model | Observable | Maximum absolute point error [dex] | Pointwise intervals within +/-0.05 dex |',
        '|---|---|---:|---:|']
    for model,label in LABELS.items():
        for observable in ('line_ratio','molecule_number_ratio','emission_per_molecule_ratio'):
            subset=[r for r in d['radial_records'] if (r['reference'],r['model'],r['observable'])==('extended',model,observable)]
            rows.append(f'| {label} | {observable} | {max(abs(r["direct_minus_surrogate"]["point"]) for r in subset):.4f} | '
                f'{sum(r["interval_within_0p05_dex"] for r in subset)}/7 |')
    rows+=['','## Uniform/off inner dip: direct adjacent contrasts','',
        'All six adjacent contrasts were reported, with within-curve Bonferroni intervals.',
        'These are the two contrasts surrounding the proposed dip, not a newly selected precision criterion.','',
        '| Reference | Contrast | Direct [dex], pointwise 95% | Within-curve familywise 95% |',
        '|---|---|---|---|']
    for r in d['inner_adjacent_contrasts']:
        if r['model']=='uniform_off' and r['observable']=='line_ratio' and r['contrast'] in (
                'annulus_2_3_minus_1_2','annulus_3_4_minus_2_3'):
            rows.append(f'| {r["reference"]} | {r["contrast"]} | {interval(r["direct"])} | {r["within_inner_curve_familywise_95"]} |')
    rows+=['','## Radius-controlled SFR slopes','',
        '| Reference | Control | Window [Myr] | Direct slope [dex/dex], 95% | Slope error, 95% |',
        '|---|---|---:|---|---|']
    for r in d['control_sfr_slopes']:
        if r['radial']=='fixed_effect':
            rows.append(f'| {r["reference"]} | {LABELS[r["model"]]} | {r["window_Myr"]} | '
                f'{interval(r["direct"])} | {interval(r["direct_minus_surrogate"])} |')
    rows+=['','## Shielding-table exposure','',
        '| Branch | Region | Line | Light outside elemental carbon-ratio table [%] | Light with frozen lookup shift >0.05 dex [%] |',
        '|---|---|---|---:|---:|']
    for r in exposure['records']:
        rows.append(f'| {r["branch"]} | {r["region"]} | {r["line"]} | '
            f'{100*r["out_of_carbon_table_light_fraction"]:.2f} | {100*r["molecular_lookup_shift_exceeds_0p05_dex_fraction"]:.2f} |')
    rows+=['','Shielding diagnostics hold the chemical state fixed. They do not test the coupled equilibrium response,',
        'bound off-table carbon ratios, or remove the fixed oxygen-table ratio of 557.',
        'Numerical failures retain unknown direct light. See JSON for missing predicted-light fractions.',
        'The paper requires a scientific review of these outputs before any conclusions are changed.']
    (ROOT/'REVIEW.md').write_text('\n'.join(rows)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True,layout='constrained')
    for ax,window in zip(axes,(1,10)):
        subset=[next(r for r in d['control_sfr_slopes'] if (r['reference'],r['model'],r['window_Myr'],r['radial'])==
            ('extended',model,window,'fixed_effect')) for model in LABELS]
        x=np.arange(4);direct=np.array([r['direct']['point'] for r in subset]);limits=np.array([r['direct']['interval_95'] for r in subset])
        ax.errorbar(x-.06,direct,yerr=[direct-limits[:,0],limits[:,1]-direct],fmt='o',capsize=3,label='Direct: common successful states')
        ax.plot(x+.06,[r['matched_surrogate'] for r in subset],'s',label='Surrogate: same weighted sample')
        ax.axhline(0,color='.5',lw=.8);ax.set_xticks(x,['Actual\non','Actual\noff','Uniform\non','Uniform\noff'])
        ax.set(title=f'{window} Myr formation window',xlabel='Isotope pattern / carbon exchange');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Radius-controlled SFR slope [dex/dex]');axes[0].legend(fontsize=8)
    fig.suptitle('x8 snapshot 575; conditional 95% sampling intervals')
    for ext in ('pdf','png'):fig.savefig(ROOT/f'control_sfr_validation.{ext}',dpi=150)
    plt.close(fig)
    (ROOT/'STATUS.md').write_text('Cached central-claim analyses and independent statistical reconstruction completed.\n'
        'Read REVIEW.md and JSON before revising the paper. Physical shielding sensitivity remains open.\n')
    print('Saved audited scientific summary and SFR validation figure.',flush=True)


if __name__=='__main__':main()
