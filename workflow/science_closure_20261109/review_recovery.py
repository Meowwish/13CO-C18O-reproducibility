"""Before/after nonresponse envelopes, including signs of scientific contrasts."""
import json
import os

import h5py
import numpy as np
from scipy.stats import norm

import closure_common as c
from prepare_recovery import cohorts
from lvg_envelope import bounds, lamda_transition, ratio_box


def main():
    assert int(os.environ.get('PBS_NCPUS', '0')) > 0
    manifest = c.verify(); values, qualified, status = c.assemble('recovery')
    with h5py.File(c.ROOT/'recovery/cases.h5') as f:
        cases = {k:f[k][:] for k in f}
    controls = cases['control'].astype(bool); good = controls & qualified
    old, new = cases['primary'][good, 6:9], values[good, 6:9]
    positive = (old > 0) & (new > 0)
    agreement = np.abs(new-old)/np.maximum(np.maximum(abs(new), abs(old)), 1e-60)
    agreement[positive] = abs(np.log10(new[positive]/old[positive]))
    transitions = [lamda_transition(c.ISO/f'chemTable_v4_full_direct/LAMDA/{name}.dat') for name in ('13co', 'c18o')]
    critical = float(norm.isf(.05/(2*10000)))
    records = []; contrasts = []; sfr_records = []
    for cohort, data, old_values, old_q in cohorts():
        n = len(data['mass']); new_values = old_values.copy(); new_q = old_q.copy()
        choose = (cases['cohort'] == cohort.encode()) & ~controls & qualified
        ids = cases['state_id'][choose]
        new_values.reshape(-1, 27)[ids] = values[choose]; new_q.ravel()[ids] = True
        c.parent.base.atomic_npz(c.ROOT/'recovery'/f'{cohort}_overlay.npz', values=new_values, qualified=new_q,
            manifest_sha256=np.asarray(manifest), complete=np.asarray(True))
        if cohort == 'global':
            kernels = np.zeros((n, 2, 19))
            source = c.ISO/'x8_claim_checks_20261105_v3'
            for part in range(4):
                with np.load(source/f'geometry/part{part:02d}.npz') as f:kernels[f['ids']] = f['kernel']
        else:
            kernels = np.zeros((n, 2, 19)); source = c.ISO/'x8_outer_validation_20261105'
            for part in range(4):
                with np.load(source/f'geometry/part{part:02d}.npz') as f:kernels[f['ids']] = f['kernel'][:, 2:4]
        for stage, v, q in [('before', old_values, old_q), ('after', new_values, new_q)]:
            boxes = {}
            for model, name in enumerate(c.MODELS):
                branch = c.MAP_BRANCH[model]; raw = data['slug'] if branch == 1 else data['uniform'] if branch >= 3 else data['actual']
                envelopes = np.stack([bounds(raw, t, iso, allow_inversion=True) for t, iso in zip(transitions, ('13CO', 'C18O'))], axis=1)
                # These are physical envelopes conditional on individual isotope
                # budgets, not a claim that every numerical solution conserves them.
                observed = v[:, branch, 7:9]; success = q[:, branch]
                lower = np.where(success[:, None], observed, envelopes[:, :, 0])
                upper = np.where(success[:, None], observed, envelopes[:, :, 1])
                wk = data['nuclei']/data['inclusion_probability']
                for r in [*range(1, 15), 15, 16, 18]:
                    w = wk*kernels[:, int(model >= 2), r]
                    zlo, zhi = w[:, None]*lower, w[:, None]*upper
                    low, high = zlo.sum(0), zhi.sum(0)
                    low95 = low-critical*np.sqrt(c.srs_variance(zlo, data['stratum'], data['population']))
                    high95 = high+critical*np.sqrt(c.srs_variance(zhi, data['stratum'], data['population']))
                    box = ratio_box(low95, high95); boxes[model, r] = box
                    records.append(dict(cohort=cohort, stage=stage, model=name, region_index=r,
                        known_qualified_states=int(success.sum()), unknown_states=int((~success).sum()),
                        design_estimated_total_lower=low, design_estimated_total_upper=high,
                        conservative_total_sampling_lower=low95, conservative_total_sampling_upper=high95,
                        log_energy_ratio_box=box))
                if cohort == 'global' and stage == 'after' and model >= 2:
                    sfr = np.zeros((n, 80))
                    for part in range(4):
                        with np.load(c.ISO/f'x8_sfr_validation_20261102/geometry/part{part:02d}.npz') as f:sfr[f['ids']] = f['kernel']
                    cell_boxes = []
                    for j in range(80):
                        w = wk*sfr[:, j]; zl, zh = w[:, None]*lower, w[:, None]*upper
                        lo = zl.sum(0)-critical*np.sqrt(c.srs_variance(zl, data['stratum'], data['population']))
                        hi = zh.sum(0)+critical*np.sqrt(c.srs_variance(zh, data['stratum'], data['population']))
                        cell_boxes.append(ratio_box(lo, hi))
                    catalogue = json.loads((c.ISO/'x8_claim_checks_20261105_v3/catalogue.json').read_text())
                    for fit in catalogue['coefficients']:
                        a = np.asarray(fit['coefficient']); used = np.flatnonzero(a != 0)
                        if any(cell_boxes[j] is None for j in used):box = None
                        else:
                            box = [sum(a[j]*cell_boxes[j][int(a[j] < 0)] for j in used),
                                   sum(a[j]*cell_boxes[j][int(a[j] >= 0)] for j in used)]
                        sfr_records.append(dict(model=name, window_Myr=fit['window_Myr'], radial=fit['radial'], slope_box=box))
            for label, a, b in [('source', 1, 0), ('exchange', 2, 3), ('uniform_exchange', 4, 5)]:
                for r in (15, 16, 18):
                    first, second = boxes[a, r], boxes[b, r]
                    box = [first[0]-second[1], first[1]-second[0]] if first is not None and second is not None else None
                    contrasts.append(dict(cohort=cohort, stage=stage, comparison=label, region_index=r, contrast_box=box))
            for model in (0, 5):
                use = [boxes[model, r] for r in range(8, 15)]
                if any(x is None for x in use):box = None
                else:
                    x = np.arange(8.5, 15.); a = (x-x.mean())/sum((x-x.mean())**2)
                    box = [sum(a[j]*use[j][int(a[j] < 0)] for j in range(7)), sum(a[j]*use[j][int(a[j] >= 0)] for j in range(7))]
                contrasts.append(dict(cohort=cohort, stage=stage, model=c.MODELS[model], comparison='outer_slope', slope_box=box))
    c.save(c.ROOT/'recovery_review.json', dict(complete=True, manifest_sha256=manifest,
        attempted=int((~controls).sum()), recovered=int((~controls & qualified).sum()),
        remaining=int((~controls & ~qualified).sum()), controls=int(controls.sum()), qualified_controls=int(good.sum()),
        maximum_control_disagreement=float(agreement.max()) if agreement.size else None,
        controls_above_0p02=int(np.any(agreement > .02, axis=1).sum()),
        envelopes=records, contrasts=contrasts, sfr_slope_envelopes=sfr_records,
        endpoint_critical=critical, endpoint_family_upper_bound=10000,
        scope='Element-constrained signed luminosity envelopes using the implemented LVG beta law, allowing inversion. They assume physical isotope budgets and normalised nonnegative level populations, fixed TCMB=2.73, no dilute IR background. Approximate design-sampling endpoint intervals, not a hydrodynamic-population census or physical-model error bound.',
        scientific_acceptance=False))
    print('Recovery and failure-envelope diagnostics complete.', flush=True)


if __name__ == '__main__':main()
