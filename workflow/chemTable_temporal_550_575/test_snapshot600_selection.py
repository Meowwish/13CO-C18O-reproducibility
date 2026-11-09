"""Frozen-model evaluation of selected statistics; cached original-600 maps only."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from Chuhan_chem_125pc import smooth_to_125pc, luminosity_to_w, LINE_FREQUENCIES_HZ
from plot_galic_selection import lattice, sample, DEPTHS

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'final_test_600/projected/snapshot600_temporal_maps.npz'
OUT = ROOT / 'final_test_600/selection'
KEYS = ('median_ratio', 'ratio_of_medians', 'ratio_of_sums')


def stats(a, b, mask):
    if mask.sum() < 10:
        return np.full(3, np.nan)
    a, b = a[mask], b[mask]
    valid = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    # A common-mask statistic is unavailable rather than silently dropping rows.
    if not valid.all():
        return np.full(3, np.nan)
    return np.array([np.median(a/b), np.median(a)/np.median(b), a.sum()/b.sum()])


def classification(a, b, s13, s18, valid):
    state = np.full(len(a), -1, dtype=np.int8)
    state[valid] = ((a[valid] > 3*s13).astype(int)
                    + 2*(b[valid] > 3*s18).astype(int))
    return state


def main():
    OUT.mkdir(exist_ok=True)
    manifest = json.loads((ROOT / 'final_test_600/frozen_manifest.json').read_text())
    model = ROOT / 'final_test_600/frozen_surrogate.joblib'
    assert hashlib.sha256(model.read_bytes()).hexdigest() == manifest['model_sha256']
    xy = lattice()
    radius = np.hypot(*xy.T)
    sector = np.floor((np.arctan2(xy[:, 1], xy[:, 0])+np.pi)/(2*np.pi)*12).astype(int) % 12
    fields = {}
    with np.load(SOURCE) as f:
        assert float(f['pixel_size_pc']) == 20
        for branch in ('direct', 'lookup'):
            fields[branch] = []
            for field, line in (('mapL13CO', '13CO'), ('mapLC18O', 'C18O')):
                w = luminosity_to_w(smooth_to_125pc(f[branch+'_'+field], 20), LINE_FREQUENCIES_HZ[line])
                fields[branch].append(sample(w, xy))
    a, b = fields['direct']
    x, y = fields['lookup']
    valid = np.isfinite(a+b+x+y) & (a >= 0) & (b >= 0) & (x >= 0) & (y >= 0)
    rows, counts = [], []
    for depth in DEPTHS:
        s13, s18 = .163/depth, .144/depth
        ds = classification(a, b, s13, s18, valid)
        ls = classification(x, y, s13, s18, valid)
        inner = valid & (radius >= 1) & (radius < 8)
        confusion = np.zeros((4, 4), dtype=int)
        np.add.at(confusion, (ds[inner], ls[inner]), 1)
        for selection in ('dual', 'dual_and_ratio_snr'):
            dm, lm = ds == 3, ls == 3
            if selection != 'dual':
                for mask, p, q in ((dm, a, b), (lm, x, y)):
                    idx = np.flatnonzero(mask)
                    mask[idx] &= (s13/p[idx])**2+(s18/q[idx])**2 < 1/9
            counts.append(dict(depth=depth, selection=selection, total=int(inner.sum()),
                               direct=int((dm & inner).sum()), lookup=int((lm & inner).sum()),
                               both=int((dm & lm & inner).sum()),
                               missed=int((dm & ~lm & inner).sum()),
                               additional=int((~dm & lm & inner).sum()),
                               confusion_dual_categories=confusion.tolist()))
            for lo in range(15):
                base = valid & (radius >= max(.5, lo)) & (radius < lo+1)
                dmask, lmask = base & dm, base & lm
                for mode, first, second in (
                    ('own_selection', dmask, lmask),
                    ('direct_selection', dmask, dmask),
                    ('intersection', dmask & lmask, dmask & lmask)):
                    dval, lval = stats(a, b, first), stats(x, y, second)
                    bias = np.log10(lval/dval)
                    jack = []
                    for k in range(12):
                        keep = sector != k
                        jack.append(np.log10(stats(x, y, second & keep)/stats(a, b, first & keep)))
                    jack = np.array(jack)
                    error = np.full(3, np.nan)
                    good = np.all(np.isfinite(jack), axis=0)
                    error[good] = np.sqrt(11/12*np.sum((jack[:, good]-jack[:, good].mean(axis=0))**2, axis=0))
                    row = dict(depth=depth, sigma13=s13, sigma18=s18, selection=selection,
                               mask_mode=mode, radius_lo=max(.5, lo), radius_hi=lo+1,
                               total=int(base.sum()), direct_count=int(first.sum()), lookup_count=int(second.sum()),
                               direct_fraction=first.sum()/base.sum() if base.any() else np.nan,
                               lookup_fraction=second.sum()/base.sum() if base.any() else np.nan)
                    for i, key in enumerate(KEYS):
                        row.update({key+'_direct':dval[i], key+'_lookup':lval[i],
                                    key+'_bias_dex':bias[i], key+'_paired_jackknife_dex':error[i]})
                    rows.append(row)
    with (OUT/'radial_statistics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for selection in ('dual', 'dual_and_ratio_snr'):
        fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True, sharey='row', layout='constrained')
        for col, mode in enumerate(('own_selection', 'direct_selection')):
            for color, depth in zip(plt.cm.plasma(np.linspace(.05, .9, len(DEPTHS))), DEPTHS):
                series = [r for r in rows if r['depth']==depth and r['selection']==selection and r['mask_mode']==mode]
                radii = [(r['radius_lo']+r['radius_hi'])/2 for r in series]
                for ax, key in zip(axes[:3, col], KEYS):
                    ax.plot(radii, [r[key+'_bias_dex'] for r in series], 'o-', ms=3, color=color, label=f'D={depth:g}')
                axes[3, col].plot(radii, [100*(r['lookup_fraction']-r['direct_fraction']) for r in series], 'o-', ms=3, color=color)
            axes[0, col].set_title(mode.replace('_', ' '))
            for ax in axes[:, col]:
                ax.axhline(0, color='.3', ls=':')
                ax.axvspan(0, 1, color='.94'); ax.axvspan(8, 15, color='.94')
                ax.set_xlim(0, 15); ax.grid(alpha=.2)
            axes[-1, col].set_xlabel('Radius [kpc]')
        for ax, label in zip(axes[:, 0], ('Median ratio bias [dex]', 'Ratio of medians bias [dex]', 'Ratio of sums bias [dex]', 'Detection fraction difference [pp]')):
            ax.set_ylabel(label)
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=7, fontsize=8)
        fig.suptitle('Original snapshot 600: frozen surrogate / direct | '+selection)
        for ext in ('png', 'pdf'):
            fig.savefig(OUT/f'selection_validation_{selection}.{ext}', dpi=180)
        plt.close(fig)
    (OUT/'summary.json').write_text(json.dumps(dict(
        source=str(SOURCE), model_sha256=manifest['model_sha256'], counts=counts,
        source_size_bytes=SOURCE.stat().st_size, beam_pc=125, sampling_pc=62.5,
        notes=['No retraining or model selection; no new DESPOTIC calculation.',
               'Full-mixture original-600 validation only; does not directly validate x8 or SLUG-only.',
               'Categories 0=neither, 1=13CO only, 2=C18O only, 3=both.',
               'R<1 and R>=8 kpc exploratory. First bin excludes R<0.5.',
               'At least 10 samples required in both estimates; unavailable statistics remain NaN in CSV.',
               'Jackknife leaves out 12 matched azimuthal sectors and measures spatial variability of bias.',
               'Direct-mask fraction difference is zero by construction.',
               'No random noise, spectral mask, new accuracy threshold, or automatic acceptance claim.']), indent=2)+'\n')
    (OUT/'README.md').write_text('# Selected-statistic validation on original snapshot 600\n\n'
        'Read summary.json for detection completeness/contamination and the four-category confusion matrix.\n'
        'radial_statistics.csv separates independent detection masks, the fixed direct mask, and their intersection.\n'
        'The figures compare independent versus fixed-direct masks; the latter isolates intensity errors from sample migration.\n'
        'Paired spatial-jackknife errors are provided in the CSV, not plotted as independent-pixel uncertainties.\n'
        'This evaluates new statistics with a previously frozen model; it must not be used for further tuning on snapshot 600.\n')
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == '__main__':
    main()
