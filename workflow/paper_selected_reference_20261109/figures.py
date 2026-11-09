"""Update two existing figures from cached summaries, without map projection."""
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parent
ISO = ROOT.parent
FIG = ISO/'Paper/figs'
STUDY = ISO/'x8_selection_reference_20261108'
CLAIMS = ISO/'x8_claim_checks_20261105_v3'
LABELS = {'actual_on':'Actual isotopes; exchange on', 'actual_off':'Actual isotopes; exchange off',
          'uniform_on':'Uniform isotopes; exchange on', 'uniform_off':'Uniform isotopes; exchange off'}
COLORS = ('#1769aa', '#bf6000', '#37803b', '#8e4d94')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    r = json.loads((STUDY/'review.json').read_text())
    a = json.loads((STUDY/'completion_review_20261109/assessment.json').read_text())
    assert a['computational_completion_passed'] and a['review_sha256'] == sha(STUDY/'review.json')
    old = ROOT/'figures_before'
    old.mkdir(exist_ok=True)
    names = ('snapshot575_exchange_shape_checked', 'elemental_to_selected_radial_outer_checked')
    for name in names:
        assert not (old/(name+'.pdf')).exists()
        shutil.copy2(FIG/(name+'.pdf'), old/(name+'.pdf'))
    checks = []
    def save(fig, name):
        fig.savefig(FIG/(name+'.pdf'))
        fig.savefig(ROOT/(name+'.png'), dpi=140)
        plt.close(fig)
    def ring(ax, x, y, color):
        ax.plot(x, y, 'o', mfc='none', mec=color, ms=9, mew=1.1)
    def own_exceptions(ax, model, curve, color):
        rows = [x for x in r['selected_statistics'] if x['model'] == model and x['selection'] == 'dual'
                and x['depth'] == 1000 and x['statistic'] == 'ratio_of_line_sums']
        for row in rows:
            j = row['annulus_kpc'][0]
            if not row['interval_within_0p05_dex']:
                ring(ax, j+.5, curve[j], color)
        checks.append(dict(model=model, selection='own_D1000', pass_annuli=[x['annulus_kpc'] for x in rows if x['interval_within_0p05_dex']]))
    # Preserve all previously shown control curves and inner direct contrasts.
    radial = json.loads((ISO/'isotope_fractionation_controls/x8_snapshot575/summary.json').read_text())['radial']
    inner = json.loads((CLAIMS/'review.json').read_text())
    inner_audit = json.loads((CLAIMS/'independent_audit.json').read_text())
    assert inner_audit['passed'] and inner_audit['review_sha256'] == sha(CLAIMS/'review.json')
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), layout='constrained')
    for ax, mode, title in zip(axes[:2], ('uncut', 'common_D1000'), ('All samples', 'Common dual detections')):
        for (model, label), color in zip(LABELS.items(), COLORS):
            rows = sorted([x for x in radial if (x['branch'], x['mode']) == (model, mode)], key=lambda x:x['r_lo'])
            ax.plot([x['r_lo']+.5 for x in rows], [x['ratio'] for x in rows], 'o-', ms=3, color=color, label=label)
            if mode == 'common_D1000':
                selected = [x for x in r['selected_chains'] if x['model'] == model and x['mode'] == 'common_controls_D1000']
                assert len(selected) == 7
                for row in selected:
                    j = row['annulus_kpc'][0]
                    if not row['interval_within_0p05_dex'][0]:
                        y = next(x['ratio'] for x in rows if x['r_lo'] == j)
                        ring(ax, j+.5, y, color)
                checks.append(dict(model=model, selection='common_controls_D1000', pass_annuli=[x['annulus_kpc'] for x in selected if x['interval_within_0p05_dex'][0]]))
        ax.axvspan(0, 1, color='.9')
        ax.axvline(8, color='.4', ls=':', lw=.8)
        ax.set(xlabel='Radius [kpc]', ylabel=r'$\sum W_{13}/\sum W_{18}$', title=title, xlim=(0, 15))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    axes[1].legend(handles=[Line2D([], [], color='.25', marker='o', mfc='none', ls='', ms=9,
        label=r'$\pm0.05$ dex interval criterion not met')], fontsize=7, loc='upper left')
    subset = [next(x for x in inner['inner_adjacent_contrasts'] if (x['reference'], x['model'], x['observable'], x['contrast']) ==
        ('extended', 'uniform_off', 'line_ratio', f'annulus_{j+1}_{j+2}_minus_{j}_{j+1}')) for j in range(1, 7)]
    x = np.arange(6)
    point = np.array([z['direct']['point'] for z in subset])
    ci = np.array([z['within_inner_curve_familywise_95'] for z in subset])
    axes[2].errorbar(x-.05, point, yerr=[point-ci[:, 0], ci[:, 1]-point], fmt='o', capsize=3, label='Direct; simultaneous within curve')
    axes[2].plot(x+.05, [z['matched_surrogate'] for z in subset], 's', label='Surrogate; matched sample', color=COLORS[1])
    axes[2].axhline(0, color='.5', lw=.8)
    axes[2].grid(axis='y', alpha=.2)
    axes[2].set(xticks=x, xticklabels=[f'{j+1}–{j+2}\n−{j}–{j+1}' for j in range(1, 7)],
        xlabel='Annulus contrast [kpc]', ylabel=r'$\Delta\log_{10}R^{13}_{18}$ [dex]', title='Uniform isotopes; exchange off')
    axes[2].legend(fontsize=6.5)
    save(fig, names[0])
    # The five original science curves are unchanged; only precision flags change.
    outer = json.loads((ISO/'x8_outer_molecular_validation_20261105/review.json').read_text())
    fig, axes = plt.subplots(5, 1, figsize=(6.6, 10.5), sharex=True, layout='constrained')
    with np.load(ISO/'highres_snapshot575_temporal/output/elemental_chain/radial_profiles.npz') as f:
        for branch, model, color in [('Full', 'production_full', COLORS[0]), ('SLUG', 'production_slug', COLORS[1])]:
            curves = f[branch]
            for ax, curve in zip(axes, curves):
                ax.plot(f['radius_kpc'], curve, 'o-', ms=3, color=color, label='Full enrichment' if branch == 'Full' else 'SLUG-only')
            for j in range(8, 15):
                row = next(x for x in outer['records'] if x['model'] == model and x['observable'] == 'molecule_number_ratio'
                    and x['region'] == f'annulus_{j}_{j+1}' and x['geometry'] == 'lattice')
                if not row['interval_within_0p05_dex']:
                    ring(axes[2], j+.5, curves[2, j], color)
            own_exceptions(axes[-1], model, curves[-1], color)
    labels = (r'Elemental $^{13}$C / $^{18}$O', r'$R_{\rm avail}$', 'Molecule-number ratio',
              'Uncut intensity ratio', r'Selected intensity ratio ($D=10^3$)')
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.set_ylabel(label, fontsize=10)
        ax.grid(alpha=.2)
        if i >= 2:
            ax.axvspan(0, 1, color='.94')
        ax.axvline(8, color='.5', ls=':', lw=.8)
    axes[0].legend(fontsize=9)
    axes[-1].legend(handles=[Line2D([], [], color='.25', marker='o', mfc='none', ls='', ms=9,
        label=r'$\pm0.05$ dex interval criterion not met')], fontsize=8, loc='upper left')
    axes[-1].set(xlabel='Radius [kpc]', xlim=(0, 15))
    save(fig, names[1])
    (ROOT/'figure_checks.json').write_text(json.dumps(dict(passed=True, checks=checks,
        outer_selected_blanket_shading_removed=True, underlying_curves_changed=False,
        new_inference_or_projection=False, figures=list(names)), indent=2)+'\n')


if __name__ == '__main__':
    main()
