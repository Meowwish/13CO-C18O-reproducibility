"""Report separately labelled resource experiments; never auto-promote science."""
from collections import Counter
import json
import os

import h5py
import numpy as np

import diagnostics_common as c
from review_sample import assess, PAIRS


def main():
    assert int(os.environ.get('PBS_NCPUS', '0')) > 0
    sha = c.verify()
    target = c.ROOT / 'review.json'
    assert not target.exists()
    with h5py.File(c.ROOT / 'cases.h5') as f:
        cases = {k: f[k][:] for k in f}
    count = len(cases['state_id'])
    state = c.base.core.empty_state(count)
    final = np.full((count, 3), -1, 'i1')
    seen = np.zeros(count, bool)
    trace_totals = Counter()
    for part in range(8):
        with np.load(c.ROOT / f'output/part{part:02d}.npz') as archive:
            f = {k: archive[k] for k in archive.files}
        ids = f['case_ids']
        assert bool(f['complete']) and not seen[ids].any()
        assert json.loads(str(f['metadata']))['manifest_sha256'] == sha
        np.testing.assert_array_equal(f['state_id'], cases['state_id'][ids])
        np.testing.assert_array_equal(f['protocol'], cases['protocol'][ids])
        assert np.all(cases['partition'][ids] == part)
        for key in state:
            state[key][ids] = f[key]
        final[ids] = f['final_returns']
        seen[ids] = True
        for raw in f['ode_trace']:
            trace = json.loads(raw.decode())
            trace_totals.update(trace['messages'])
    assert seen.all() and np.all(state['status'] != 0)
    assert np.all(np.isnan(state['values'][state['status'] < 0]))
    admissible = c.admissible(state['status'], final)
    is_target = cases['kind'] == 0
    with np.load(c.PARENT / 'candidate_overlay.npz') as f:
        parent_values, parent_qualified, parent_origin = (f[k].copy() for k in ('values', 'qualified', 'origin'))
    n = len(parent_qualified)
    summaries, overlays = {}, {}
    errors = c.agreement_error(cases['previous_values'][:, 6:9], state['values'][:, 6:9])
    for protocol, label in enumerate(c.PROTOCOLS):
        pick = cases['protocol'] == protocol
        controls = pick & ~is_target
        good_controls = controls & admissible
        summaries[label] = dict(target_cases=int((pick & is_target).sum()),
            target_nominal_success=int((pick & is_target & (state['status'] == 1)).sum()),
            recovered_targets=int((pick & is_target & admissible).sum()),
            nominal_success_final_emitter_failure=int((pick & (state['status'] == 1) & ~admissible).sum()),
            control_cases=int(controls.sum()), control_success=int(good_controls.sum()),
            max_control_line_error=float(errors[good_controls].max()) if good_controls.any() else None,
            controls_exceeding_0p02=int(np.sum(good_controls & np.any(errors > .02, axis=1))),
            failure_errors=dict(Counter(state['error'][pick & ~admissible].astype(str))))
        v, q = parent_values.copy(), parent_qualified.copy()
        choose = pick & is_target & admissible
        tid = cases['state_id'][choose]
        v.reshape(n*5, 27)[tid] = state['values'][choose]
        q.ravel()[tid] = True
        overlays[label] = (v, q)
    # No agreement gate: report all A/B comparisons for later scientific review.
    np.testing.assert_array_equal(cases['state_id'][::2], cases['state_id'][1::2])
    both = admissible[::2] & admissible[1::2]
    paired_error = c.agreement_error(state['values'][::2, 6:9], state['values'][1::2, 6:9])
    summaries['protocol_comparison'] = dict(both_success=int(both.sum()),
        max_line_disagreement=float(paired_error[both].max()) if both.any() else None,
        pairs_exceeding_0p02=int(np.sum(both & np.any(paired_error > .02, axis=1))))
    v, q = parent_values.copy(), parent_qualified.copy()
    origin = parent_origin.copy()
    # Apply B then A, ensuring A has preference when both qualify.
    for protocol in (1, 0):
        choose = (cases['protocol'] == protocol) & is_target & admissible
        tid = cases['state_id'][choose]
        v.reshape(n*5, 27)[tid] = state['values'][choose]
        q.ravel()[tid] = True
        origin.ravel()[tid] = 4 + protocol
    overlays['combined_A_preferred'] = (v, q)
    c.base.atomic_npz(c.ROOT / 'diagnostic_candidates.npz', manifest_sha256=np.asarray(sha), values=v,
                      qualified=q, origin=origin, case_state_id=cases['state_id'], case_protocol=cases['protocol'],
                      case_status=state['status'], final_returns=final, admissible=admissible)
    with h5py.File(c.SOURCE / 'sample.h5') as f:
        data = {k: f['sample/' + k][:] for k in ('mass', 'nuclei', 'xyz', 'inclusion_probability', 'stratum')}
        proxy, pop, sn = (f[k][:] for k in ('sample/lum_per_H', 'stratum_population', 'stratum_sample'))
        exact = {g: f['predicted_' + g + '_totals'][:] for g in ('particle', 'beam')}
    kernel = np.full((n, 9), np.nan)
    for part in range(4):
        with np.load(c.SOURCE / f'geometry/part{part:02d}.npz') as f:
            kernel[f['ids']] = f['kernel_weights']
    freq = json.loads((c.ISO / 'highres_snapshot575_temporal/config.json').read_text())['projection']['line_frequencies_Hz']
    offset = 3*np.log10(freq['C18O']/freq['13CO'])
    records = []
    for label, (v, q) in overlays.items():
        summaries.setdefault(label, {})['unresolved_states'] = int((~q).sum())
        direct, valid = v[:, c.base.MODEL_BRANCH, 6:9], q[:, c.base.MODEL_BRANCH]
        for geometry, g in (('particle', c.base.region_masks(data['xyz']).astype(float)), ('beam', kernel)):
            for r, region in enumerate(c.base.REGIONS):
                weight = data['nuclei']/data['inclusion_probability']*g[:, r]
                mass = data['mass']/data['inclusion_probability']*g[:, r]
                comparisons = [(c.base.MODELS[i], [i]) for i in range(6)] + list(PAIRS.items())
                for name, models in comparisons:
                    value = assess(list(models), direct, proxy, valid, weight, mass, data['stratum'], pop, sn, exact[geometry][r], offset)
                    records.append(dict(protocol=label, geometry=geometry, region=region, comparison=name, **value))
    combined_qualified = overlays['combined_A_preferred'][1]
    c.save(target, dict(complete=True, manifest_sha256=sha, protocols=summaries,
        combined_newly_recovered_states=int(combined_qualified.sum()-parent_qualified.sum()),
        combined_unresolved_states=int((~combined_qualified).sum()), ode_messages=dict(trace_totals), results=records,
        scientific_acceptance=False, manuscript_acceptance=False,
        note='Resource diagnostics only. All population and physical-model qualifications remain; no automatic promotion or additional job.'))
    text = ['# Fixed numerical-resource diagnostic completed', '',
            'Original labels and preceding overlay are unchanged. Candidate protocols require final emitter success.', '',
            '| Protocol | Recovered targets | Qualified controls | Controls above 0.02 |',
            '|---|---:|---:|---:|']
    for label in c.PROTOCOLS:
        r = summaries[label]
        text.append(f"| {label} | {r['recovered_targets']} | {r['control_success']}/160 | {r['controls_exceeding_0p02']} |")
    text += ['', 'A/B comparisons, remaining masks and conditional survey estimates are in review.json.',
             'No automatic scientific acceptance or further retry.', '']
    (c.ROOT / 'RESULTS.md').write_text('\n'.join(text))
    print(json.dumps(c.base.clean(summaries), indent=2), flush=True)


if __name__ == '__main__':
    main()
