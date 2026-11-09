"""Redraw only cached quantities, replacing blanket outer-disc shading."""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
ISO = ROOT.parent
FIG = ISO/'Paper/figs'
INNER = ISO/'x8_claim_checks_20261105_v3'
OUTER = ISO/'x8_outer_molecular_validation_20261105'
sys.path.extend([str(ISO), str(ISO/'highres_snapshot575_temporal')])
from plot_galic_selection import lattice, sample

LABELS = {'actual_on':'Actual; exchange on', 'actual_off':'Actual; exchange off',
    'uniform_on':'Uniform; exchange on', 'uniform_off':'Uniform; exchange off'}
COLORS = ('#1769aa', '#bf6000', '#37803b', '#8e4d94')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(fig, name):
    fig.savefig(FIG/(name+'.pdf')); fig.savefig(ROOT/(name+'.png'), dpi=140); plt.close(fig)


def main():
    assert json.loads((ROOT/'shielding_audit.json').read_text())['passed']
    inner = json.loads((INNER/'review.json').read_text()); outer = json.loads((OUTER/'review.json').read_text())
    for directory in (INNER, OUTER):
        audit = json.loads((directory/'independent_audit.json').read_text())
        assert audit['passed'] and audit['review_sha256'] == sha(directory/'review.json')
    original = json.loads((ISO/'x8_outer_validation_20261105/manifest.json').read_text())
    for path in (ISO/'isotope_fractionation_controls/x8_snapshot575/maps_125pc.npz',
                 ISO/'highres_snapshot575_temporal/output/snapshot_575_maps_125pc.npz',
                 ISO/'highres_snapshot575_temporal/slug_only/output/snapshot_575_maps_125pc.npz'):
        assert sha(path) == original['hashes'][str(path)]
    def record(model, observable, radial):
        rows = inner['radial_records'] if radial < 8 else outer['records']
        return next(x for x in rows if x['model'] == model and x['observable'] == observable
            and x['region'] == f'annulus_{radial}_{radial+1}'
            and (x.get('reference') == 'extended' if radial < 8 else x.get('geometry') == 'lattice'))
    def exceptions(ax, model, observable, curve, color):
        bad = [j for j in range(8, 15) if not record(model, observable, j)['interval_within_0p05_dex']]
        if bad: ax.plot(np.array(bad)+.5, np.asarray(curve)[bad], 'o', mfc='none', mec=color, ms=9, mew=1.1)
    frequency = json.loads((ISO/'highres_snapshot575_temporal/config.json').read_text())['projection']['line_frequencies_Hz']
    offset = 3*np.log10(frequency['C18O']/frequency['13CO'])
    # Main chain: elemental inputs need no surrogate validation; the uncut
    # molecular and line stages now have outer references. Only selection
    # retains outer shading until its own direct-reference review completes.
    fig, axes = plt.subplots(5, 1, figsize=(6.6, 10.5), sharex=True, layout='constrained')
    with np.load(ISO/'highres_snapshot575_temporal/output/elemental_chain/radial_profiles.npz') as f:
        for branch, model, color in [('Full', 'production_full', COLORS[0]), ('SLUG', 'production_slug', COLORS[1])]:
            curves = f[branch]
            for ax, curve in zip(axes, curves): ax.plot(f['radius_kpc'], curve, 'o-', ms=3, color=color, label='Full enrichment' if branch == 'Full' else 'SLUG-only')
            exceptions(axes[2], model, 'molecule_number_ratio', curves[2], color)
    labels = (r'Elemental $^{13}$C / $^{18}$O', r'$R_{\rm avail}$', 'Molecule-number ratio', 'Uncut intensity ratio', r'Selected intensity ratio ($D=10^3$)')
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.set_ylabel(label, fontsize=10); ax.grid(alpha=.2)
        if i >= 2: ax.axvspan(0, 1, color='.94')
        if i == 4: ax.axvspan(8, 15, color='.92')
        ax.axvline(8, color='.5', ls=':', lw=.8)
    axes[0].legend(fontsize=9); axes[-1].set(xlabel='Radius [kpc]', xlim=(0, 15))
    save(fig, 'elemental_to_selected_radial_outer_checked')
    def matched(ax, model, observable, color, source_label=None):
        rows = [record(model, observable, r) for r in range(1, 15)]
        shift = 0 if observable == 'molecule_number_ratio' else offset
        point = np.array([x['direct_log_ratio']['point']+shift for x in rows]); ci = np.array([x['direct_log_ratio']['interval_95'] for x in rows])+shift
        y = 10**point
        ax.errorbar(np.arange(1.5, 15), y, yerr=[y-10**ci[:, 0], 10**ci[:, 1]-y], fmt='o-', ms=3, capsize=2, color=color, label=source_label)
        proxy = 10**np.array([x['matched_surrogate_log_ratio']+shift for x in rows])
        ax.plot(np.arange(1.5, 15), proxy, 's--', ms=2.5, color=color, alpha=.7)
        bad = np.array([not x['interval_within_0p05_dex'] for x in rows])
        ax.plot(np.arange(1.5, 15)[bad], y[bad], 'o', mfc='none', mec=color, ms=8)
        ax.axvline(8, color='.5', ls=':', lw=.8); ax.grid(alpha=.2); ax.set_xlim(1, 15)
    # Full / SLUG appendix: retain a single-column, two-row layout.
    fig, axes = plt.subplots(2, 1, figsize=(4.1, 6.1), sharex=True, layout='constrained')
    for model, label, color in [('production_full', 'Full enrichment', COLORS[0]), ('production_slug', 'SLUG-only', COLORS[1])]:
        matched(axes[0], model, 'molecule_number_ratio', color, label)
        matched(axes[1], model, 'line_ratio', color, label)
    axes[0].set_ylabel('Molecule-number ratio', fontsize=9); axes[1].set_ylabel('Intensity ratio', fontsize=9)
    axes[0].legend(fontsize=7); axes[1].set_xlabel('Radius [kpc]', fontsize=9)
    for ax in axes: ax.tick_params(labelsize=8)
    save(fig, 'molecular_to_intensity_radial_outer_checked')
    # Representative decomposition references now cover all tested radii.
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, layout='constrained')
    for col, (model, title) in enumerate([('production_full', 'Full enrichment'), ('uniform_off', 'Uniform isotopes; exchange off')]):
        matched(axes[0, col], model, 'molecule_number_ratio', COLORS[0])
        matched(axes[1, col], model, 'emission_per_molecule_ratio', COLORS[0])
        axes[0, col].set_title(title, fontsize=11); axes[1, col].set_xlabel('Radius [kpc]')
        axes[0, col].set_ylabel('Molecule-number ratio'); axes[1, col].set_ylabel('Relative emission per molecule')
    from matplotlib.lines import Line2D
    axes[0, 0].legend(handles=[Line2D([0], [0], color=COLORS[0], marker='o', label='Direct: 95% sampling interval'),
        Line2D([0], [0], color=COLORS[0], marker='s', ls='--', alpha=.7, label='Surrogate: matched population'),
        Line2D([0], [0], color=COLORS[0], marker='o', mfc='none', ls='', ms=8, label=r'$\pm0.05$ dex criterion not met')], fontsize=7)
    save(fig, 'snapshot575_molecular_reference_outer_checked')
    # Regenerate controls from cached beam maps, with the exact old selection.
    xy = lattice(); radius = np.hypot(*xy.T); fields = {}
    with np.load(ISO/'isotope_fractionation_controls/x8_snapshot575/maps_125pc.npz') as f:
        for model in LABELS:
            fields[model] = {k: sample(f[model+'_'+k], xy) for k in ('W13CO', 'WC18O', 'X13CO', 'XC18O')}
    common = np.all([(x['W13CO'] > 3*.000163)&(x['WC18O'] > 3*.000144) for x in fields.values()], axis=0)
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True, layout='constrained')
    for col, selected in enumerate((np.ones(len(radius), bool), common)):
        for (model, label), color in zip(LABELS.items(), COLORS):
            curves = np.full((3, 15), np.nan); v = fields[model]
            for j in range(15):
                use = (radius >= j)&(radius < j+1)&selected
                if use.sum() < 10: continue
                m = v['X13CO'][use].sum()/v['XC18O'][use].sum()
                l = v['W13CO'][use].sum()/v['WC18O'][use].sum()
                curves[:, j] = m, l, l/m
            for ax, curve in zip(axes[:, col], curves): ax.plot(np.arange(15)+.5, curve, 'o-', color=color, label=label, ms=3)
            if col == 0:
                for row, observable in enumerate(('molecule_number_ratio', 'line_ratio', 'emission_per_molecule_ratio')):
                    exceptions(axes[row, col], model, observable, curves[row], color)
        axes[0, col].set_title('All samples' if col == 0 else 'Common dual detections', fontsize=11)
        for ax in axes[:, col]:
            ax.axvspan(0, 1, color='.94'); ax.axvline(8, color='.5', ls=':', lw=.8); ax.grid(alpha=.2); ax.set_xlim(0, 15)
            if col: ax.axvspan(8, 15, color='.92')
        axes[2, col].axhline(1, color='.5', ls='-.', lw=.8); axes[2, col].set_xlabel('Radius [kpc]')
    for row, label in enumerate(('Molecule-number ratio', 'Intensity ratio', 'Relative emission per molecule')):
        for ax in axes[row]: ax.set_ylabel(label)
    axes[0, 0].legend(fontsize=7)
    save(fig, 'snapshot575_x8_molecular_emission_outer_checked')
    report = dict(passed=True, outer_review_sha256=sha(OUTER/'review.json'),
        inner_review_sha256=sha(INNER/'review.json'), redrawn_uncut_outer_shading_removed=True,
        selected_shading_retained_pending_own_validation=True, new_surrogate_queries=0,
        files=['elemental_to_selected_radial_outer_checked', 'molecular_to_intensity_radial_outer_checked',
            'snapshot575_molecular_reference_outer_checked', 'snapshot575_x8_molecular_emission_outer_checked'])
    (ROOT/'figure_checks.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Four figures redrawn from audited caches.', flush=True)


if __name__ == '__main__': main()
