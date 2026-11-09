"""Readable audited outer-disc results and a standalone validation figure."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
LABELS = dict(production_full='Full enrichment (production)', production_slug='SLUG-only',
              actual_on='Actual isotope ratios, exchange on (paired model)',
              actual_off='Actual isotope ratios, exchange off',
              uniform_on='Uniform isotope ratios, exchange on',
              uniform_off='Uniform isotope ratios, exchange off')


def main():
    audit = json.loads((ROOT/'independent_audit.json').read_text())
    assert audit['passed']
    d = json.loads((ROOT/'review.json').read_text())
    offset = d['ratio_offset_energy_to_brightness_dex']
    def outer(row):
        return row['region'].startswith('annulus_') and 8 <= int(row['region'].split('_')[1]) < 15
    rows = ['# Outer-disc validation: scientific review', '',
        'All 14 mk27/rsaa jobs finished with exit status 0. The independently checked',
        'reference supports an overall outward rise over 8--15 kpc in the tested',
        'equilibrium model, including with uniform isotope ratios and exchange off.',
        'This is not a claim of physical-model accuracy, strict monotonicity,',
        'agreement with an observed gradient, or inside-out galaxy growth.', '',
        '## Computation and evidence', '',
        f'- Independent sample: {d["independent_sample_particles"]:,} particles; 4096 in each outer 1-kpc annulus.',
        f'- Qualified: {audit["qualified_states"]:,}/{audit["physical_states"]:,} states '
        f'({100*audit["qualified_states"]/audit["physical_states"]:.2f}%).',
        f'- Unresolved: {audit["failed_states"]:,} failed states and {audit["timed_out_states"]:,} timeouts; '
        f'{audit["reused_states"]:,} existing qualified labels reused.',
        '- The reference and the surrogate are compared on the same successful states with inverse inclusion probabilities.',
        '- Five physical states share six surrogate predictions; production Full and paired actual/on share their direct reference.',
        f'- Independent reconstruction: {audit["interval_sets_independently_reconstructed"]} interval sets, '
        f'maximum point/SE differences {audit["maximum_point_difference"]:.3g}/{audit["maximum_standard_error_difference"]:.3g}.',
        '- Native-pixel and 62.5 pc lattice sums agree on the conclusions; each uses its actual smoothing operator.',
        f'- Service usage: {audit["service_units"]:.2f} SU = {audit["MSU"]:.8f} MSU.',
        '- Direct partitions ran for approximately 2 h 4 min; preparation-to-final-review elapsed approximately 2 h 17 min, excluding initial queue time.', '',
        '## Lattice-sampled outer-disc precision and trend', '',
        'Intervals below are nominal 95% stratified sampling intervals conditional on successful states.', '',
        '| Branch/model | Largest absolute annular ratio error [dex] | Pointwise error intervals inside +/-0.05 dex | Direct outer slope [dex/kpc] | 95% interval |',
        '|---|---:|---:|---:|---|']
    for model, label in LABELS.items():
        subset = [r for r in d['model_checks'] if r['geometry']=='lattice' and r['model']==model and outer(r)]
        slope = next(r['direct'] for r in d['shape_checks'] if r['geometry']=='lattice' and r['model']==model
                     and r['contrast']=='outer_OLS_slope_dex_per_kpc')
        rows.append(f'| {label} | {max(abs(r["direct_minus_surrogate_dex"]["point"]) for r in subset):.5f} | '
                    f'{sum(r["interval_within_0p05_dex"] for r in subset)}/7 | {slope["point"]:.5f} | '
                    f'[{slope["interval_95"][0]:.5f}, {slope["interval_95"][1]:.5f}] |')
    rows += ['', 'The seven Full pointwise intervals are within the 0.05 dex screen. This is not a simultaneous full-population error bound.', '',
        '## Qualified direct outer profile', '',
        '| Annulus [kpc] | Full | Uniform isotope ratios, exchange off |',
        '|---|---:|---:|']
    for r in range(8, 15):
        region = f'annulus_{r}_{r+1}'
        values = [next(v for v in d['model_checks'] if v['geometry']=='lattice' and v['model']==m
                       and v['region']==region)['direct_log_energy_ratio']['point'] for m in ('production_full', 'uniform_off')]
        rows.append(f'| {r}--{r+1} | {10**(values[0]+offset):.3f} | {10**(values[1]+offset):.3f} |')
    rows += ['', 'These are probability-weighted conditional reference estimates, not direct solutions for every particle.', '',
        '## What is supported and what is not', '',
        '- The broad outer increase survives direct calculation. Full rises by about a factor of two between the first and last outer annuli.',
        '- Uniform/off also retains a positive outer slope and positive endpoint contrast. Spatial isotope variation and carbon exchange are not necessary for this outer rise in this model.',
        '- Adjacent annuli fluctuate; a positive fitted slope does not imply a strictly monotonic profile.',
        '- No observation-matched absolute intensity, selection test, or inside-out growth inference is established.', '',
        '## Precision exceptions', '',
        'Uniform/off at 14--15 kpc has direct-minus-surrogate error -0.04245 dex',
        'with 95% interval [-0.05146, -0.03345] on the lattice. Its error interval',
        'extends slightly outside the 0.05 dex screen; the native-pixel comparison',
        'also fails that screen. Do not call every outer branch precision-qualified.', '',
        'A shape statistic needs its own check: the Uniform/off endpoint contrast',
        'has an error of -0.05449 dex (95% interval [-0.06632, -0.04265]). The surrogate',
        'therefore overstates its outer rise, even though the directly evaluated rise',
        'remains positive. Other paired controls and SLUG-only also show slope biases.',
        'Additional random draws alone do not remove a surrogate shape bias.', '',
        '## Outer integrated paired controls', '',
        '| Intervention | Direct response [%] | Conditional 95% interval [%] |',
        '|---|---:|---|']
    for name, label in [('source','Remove old-population source contributions'),
                        ('exchange','Enable exchange, actual isotope ratios'),
                        ('uniform_exchange','Enable exchange, uniform isotope ratios')]:
        r = next(r for r in d['paired_checks'] if r['geometry']=='lattice' and r['comparison']==name and r['region']=='outer_8_15')
        q = r['direct_response_dex']
        point, lo, hi = 100*(10**np.array([q['point'], *q['interval_95']])-1)
        rows.append(f'| {label} | {point:+.2f} | [{lo:+.2f}, {hi:+.2f}] |')
    maximum_missing = max(max(r['missing_predicted_positive_line_fractions']) for r in d['model_checks'] if outer(r))
    rows += ['', f'Across these outer annular model checks, unresolved predicted positive-line light reaches {100*maximum_missing:.2f}%.',
        'This is not an upper bound on unknown direct emission. Intervals omit unresolved-state, physical-model and spatial-model uncertainty.', '',
        '## Consequence for the paper', '',
        'The previous 1--8 kpc restriction can no longer be justified solely by a lack',
        'of outer direct checks. Keep the full radial range, distinguish inner-disc',
        'non-monotonic structure from the broad outer increase, and explicitly state',
        'the conditional direct support and the Uniform/off amplitude limitation.',
        'Do not retune the frozen model on these validation labels and then reuse',
        'the same labels as an independent test. No manuscript or GitHub change is',
        'performed by this result review.', '',
        'outer_validation.pdf/png compares the matched successful-state direct and',
        'surrogate curves. Error bars quantify sampling, not total model uncertainty.']
    (ROOT/'REVIEW.md').write_text('\n'.join(rows)+'\n')
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, layout='constrained',
                             gridspec_kw={'height_ratios': [1.5, 1.]})
    for col, (model, title) in enumerate([('production_full','Full enrichment'),
                                         ('uniform_off','Uniform isotopes; exchange off')]):
        subset = [next(r for r in d['model_checks'] if r['geometry']=='lattice' and r['model']==model
                       and r['region']==f'annulus_{i}_{i+1}') for i in range(8, 15)]
        x = np.arange(8.5, 15.)
        direct = np.array([r['direct_log_energy_ratio']['point'] for r in subset])
        limits = np.array([r['direct_log_energy_ratio']['interval_95'] for r in subset])
        y = 10**(direct+offset)
        low, high = 10**(limits[:, 0]+offset), 10**(limits[:, 1]+offset)
        p = 10**(np.array([r['matched_surrogate_log_energy_ratio'] for r in subset])+offset)
        axes[0, col].errorbar(x, y, yerr=[y-low, high-y], fmt='o-', capsize=3,
                             label='Direct reference, successful states', color='#1769aa')
        axes[0, col].plot(x, p, 's--', color='#bf6000', label='Surrogate, same weighted sample')
        axes[0, col].set(title=title, ylabel=r'$\sum W_{13}/\sum W_{18}$')
        axes[0, col].legend(fontsize=8)
        error = np.array([r['direct_minus_surrogate_dex']['point'] for r in subset])
        ci = np.array([r['direct_minus_surrogate_dex']['interval_95'] for r in subset])
        axes[1, col].axhspan(-.05, .05, color='#dbead5', label='0.05 dex screening range')
        axes[1, col].axhline(0, color='.5', lw=.8)
        axes[1, col].errorbar(x, error, yerr=[error-ci[:, 0], ci[:, 1]-error], fmt='o-',
                             color='#1769aa', capsize=3)
        axes[1, col].set(xlabel='Radius [kpc]', ylabel='Direct minus surrogate [dex]', ylim=(-.07, .065))
        for ax in axes[:, col]:
            ax.set_xlim(8, 15)
            ax.grid(alpha=.2)
    fig.suptitle('x8 snapshot 575: independent outer-disc reference\n125 pc beam; 62.5 pc sampling; conditional 95% sampling intervals', fontsize=11)
    for extension in ('png', 'pdf'):
        fig.savefig(ROOT/f'outer_validation.{extension}', dpi=180)
    plt.close(fig)
    print('Saved scientific review and outer_validation.pdf/png.')


if __name__ == '__main__':
    main()
