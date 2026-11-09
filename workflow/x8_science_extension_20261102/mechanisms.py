"""Existing solutions only: luminosity-weighted actual tau/beta/Tex distributions."""
import os
import numpy as np
import extension as c


def quantiles(x, w):
    valid = np.isfinite(x)&np.isfinite(w)&(w>0)
    if not valid.any(): return [None]*3
    order = np.argsort(x[valid]); x = x[valid][order]; w = w[valid][order]
    return np.interp([.1,.5,.9], (np.cumsum(w)-.5*w)/w.sum(), x).tolist()


def main():
    assert int(os.environ.get('PBS_NCPUS','0')) > 0
    sha = c.verify(); data, population, geometry = c.sample_data()
    records, checks = [], []
    for label, path in (('default_reference',c.ISO/'x8_reference_recovery_20261101/candidate_overlay.npz'),
                        ('resource_extended',c.ISO/'x8_failure_diagnostics_20261102/diagnostic_candidates.npz')):
        with np.load(path) as f:
            values, qualified = f['values'], f['qualified']
        for b, branch in enumerate(c.BRANCHES):
            q = qualified[:,b]
            for j, line in enumerate(('CO','13CO','C18O')):
                tau, beta, tex, lum = (values[:,b,k+j] for k in (9,18,12,6))
                predicted = np.full(len(tau),np.nan)
                normal = q&(tau>1e-6)
                predicted[normal] = -np.expm1(-tau[normal])/tau[normal]
                predicted[q&~normal] = 1-tau[q&~normal]/2
                checks.append(dict(protocol=label,branch=branch,line=line,negative_tau=int((q&(tau<0)).sum()),
                                   maximum_beta_formula_difference=float(np.max(np.abs(beta[q]-predicted[q])))))
                for r, region in enumerate(c.REGIONS):
                    base = data['nuclei']/data['inclusion_probability']*geometry[:,r]
                    weight = np.zeros(len(tau)); weight[q] = base[q]*np.maximum(lum[q],0)
                    total = weight.sum()
                    if total <= 0: continue
                    fractions = {}
                    for name, use in (('tau_gt_1',tau>1),('beta_lt_0.5',beta<.5),('tau_negative',tau<0)):
                        point = (weight*use).sum()/total
                        fractions[name] = c.interval(point, weight*(use-point)/total,
                                                     data['stratum'],population)
                    signed = np.sum(base[q]*lum[q])
                    proxy = base*data['lum_per_H'][:,[2,1,3,4,5][b],j]
                    missing = proxy[~q].sum()/proxy.sum() if proxy.sum()>0 else None
                    records.append(dict(protocol=label,branch=branch,line=line,region=region,
                        tau_p10_median_p90=quantiles(tau,weight),beta_p10_median_p90=quantiles(beta,weight),
                        Tex_K_p10_median_p90=quantiles(tex,weight),Tg_K_p10_median_p90=quantiles(values[:,b,0],weight),
                        positive_emission_HT_total=total,signed_HT_total=signed,
                        known_fiducial_predicted_missing_light_fraction=missing,
                        effective_luminous_particles=float(total**2/np.sum(weight**2)),
                        largest_luminosity_weight_fraction=float(weight.max()/total),fractions=fractions))
            # Same positive C18O-emission weighting for both differential quantities.
            for r, region in enumerate(c.REGIONS):
                w = np.zeros(len(q)); valid = q&(values[:,b,20]>0)&(values[:,b,14]>0)
                w[valid] = data['nuclei'][valid]/data['inclusion_probability'][valid]*geometry[valid,r]*np.maximum(values[valid,b,8],0)
                br = np.full(len(q),np.nan); tr = br.copy()
                br[valid] = values[valid,b,19]/values[valid,b,20]
                tr[valid] = values[valid,b,13]/values[valid,b,14]
                records.append(dict(protocol=label,branch=branch,region=region,
                    weighting='positive C18O emission on qualified positive beta18 and Tex18 states',
                    beta13_over_beta18_p10_median_p90=quantiles(br,w),
                    Tex13_over_Tex18_p10_median_p90=quantiles(tr,w)))
    c.save(c.ROOT/'mechanism_review.json',dict(manifest_sha256=sha,records=records,beta_consistency=checks,
        scope='Actual per-cloud LVG outputs; luminosity-weighted distributions use positive emission, signed luminosity totals retained. Conditional on qualified states.',
        limitations=['Not projected galaxy optical depth.','No product of averaged factors or causal opacity percentage.',
                     'Negative-tau beta uses the existing runtime linear fallback; not a maser calculation.']))
    (c.ROOT/'MECHANISM_REVIEW.md').write_text('# Existing-solution mechanism diagnostics\n\n'
        'Actual stored LVG escape probabilities, optical depths and excitation temperatures are summarized in '
        'mechanism_review.json for both reference protocols and all nine apertures. Positive-emission '
        'distributions and signed totals are distinguished. These diagnostics do not assign a causal opacity fraction.\n')
    print('Mechanism diagnostics complete; no new chemistry.')


if __name__ == '__main__': main()
