"""Frozen-model intervention audit, not a new snapshot-600 final test."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from adaptive_local_surrogate import AdaptiveLocalActualAbundanceSurrogate

ROOT = Path(__file__).resolve().parent
ISO = ROOT.parent
LINES = ('L_CO_1_0', 'L_13CO_1_0', 'L_C18O_1_0')


def weighted_quantile(x, w, q):
    use = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not use.any():
        return None
    order = np.argsort(x[use])
    cumulative = np.cumsum(w[use][order])
    return float(np.interp(q*cumulative[-1], cumulative, x[use][order]))


def main():
    out = ROOT / 'output_expanded/uniform_validation'
    out.mkdir(exist_ok=True)
    frozen = ISO / 'chemTable_temporal_550_575/final_test_600'
    manifest = json.loads((frozen/'frozen_manifest.json').read_text())
    digest = hashlib.sha256((frozen/'frozen_surrogate.joblib').read_bytes()).hexdigest()
    assert digest == manifest['model_sha256']
    model = AdaptiveLocalActualAbundanceSurrogate(frozen/'frozen_surrogate.joblib', require_validated=False)
    reports = []
    for number in (550,575):
        path = ROOT/f'output_expanded/snapshot{number}_paired_retry.h5'
        with h5py.File(path) as f:
            assert bool(f.attrs['complete'])
            names = json.loads(f.attrs['output_names_json'])
            mass = f['mass_Msun'][:]
            weight = mass / f['inclusion_probability'][:]
            source_rows = f['source_row'][:]
            particle_ids = f['particle_index'][:]
            radius = f['radius_kpc'][:]
            good = np.logical_and.reduce([f[b+'/status'][:] == 1 for b in
                ('actual_on','actual_off','uniform_on','uniform_off')])
            actual_status = f['actual_on/status'][:] == 1
            report = dict(snapshot=number, selected=len(good), common_converged=int(good.sum()), branches={})
            saved = dict(source_rows=source_rows, radius_kpc=radius, weight=weight, common_converged=good)
            for branch, key in (('actual_on','actual_inputs'),('uniform_on','uniform_inputs')):
                raw = f[key][:]
                prediction, diagnostic = model.query(raw, chunk_size=10000)
                values = dict(zip(names,f[branch+'/values'][:].T))
                covered = diagnostic['within_axis_bounds'] & diagnostic['within_joint_coverage']
                record = dict(weighted_mass_covered=float(weight[covered].sum()/weight.sum()), lines={}, radial=[])
                biases = []
                for name in LINES:
                    direct = values[name]
                    pred = prediction[name]
                    valid = good & np.isfinite(direct) & (direct >= 0) & np.isfinite(pred) & (pred >= 0)
                    positive = valid & (direct > 0) & (pred > 0)
                    err = np.log10(pred[positive]/direct[positive])
                    luminosity_weight = weight[positive]*direct[positive]
                    ds, ps = np.sum(weight[valid]*direct[valid]), np.sum(weight[valid]*pred[valid])
                    bias = float(np.log10(ps/ds)) if min(ds,ps)>0 else None
                    biases.append(bias)
                    record['lines'][name] = dict(valid_pairs=int(valid.sum()), sum_bias_dex=bias,
                        direct_luminosity_weighted_p95_abs_dex=weighted_quantile(abs(err),luminosity_weight,.95),
                        direct_luminosity_covered=float(np.sum(weight[valid & covered]*direct[valid & covered])/ds) if ds>0 else None,
                        zero_prediction_positive_reference=int(np.sum(valid & (direct>0) & (pred==0))))
                    for lo in range(1,8):
                        use = valid & (radius>=lo) & (radius<lo+1)
                        dsum = np.sum(weight[use]*direct[use]); psum=np.sum(weight[use]*pred[use])
                        record['radial'].append(dict(line=name,r_lo=lo,count=int(use.sum()),
                            bias_dex=float(np.log10(psum/dsum)) if min(dsum,psum)>0 else None))
                    saved[branch+'_direct_'+name] = direct
                    saved[branch+'_predicted_'+name] = pred
                record['summed_rare_ratio_bias_dex'] = biases[1]-biases[2] if None not in biases[1:] else None
                record['total_bias_screen_passed'] = all(b is not None and abs(b)<=.05 for b in biases) and abs(record['summed_rare_ratio_bias_dex'])<=.05
                report['branches'][branch] = record
            reference = ISO/f'chemTable_temporal_550_575/snapshot{number}/output/snapshot_{number}_direct_merged.h5'
            order = np.argsort(source_rows)
            inverse = np.argsort(order)
            with h5py.File(reference) as ref:
                assert bool(ref.attrs['complete'])
                np.testing.assert_array_equal(ref['particle_index'][source_rows[order]][inverse],particle_ids)
                ref_good = ref['status'][source_rows[order]][inverse]>0
                report['missing_reference'] = dict(unknown_reference_rows=int(np.sum(~good & ~ref_good)),lines={})
                for name in LINES:
                    ref_value=ref[name][source_rows[order]][inverse]
                    valid=ref_good & np.isfinite(ref_value)
                    w=weight*np.maximum(np.nan_to_num(ref_value),0)
                    total=w[valid].sum()
                    report['missing_reference']['lines'][name] = dict(
                        common_failure_baseline_luminosity_fraction=float(w[valid & ~good].sum()/total) if total>0 else None,
                        actual_failure_baseline_luminosity_fraction=float(w[valid & ~actual_status].sum()/total) if total>0 else None,
                        caveat='Actual/on baseline only; does not bound exchange-off or uniform missing luminosity.')
            np.savez_compressed(out/f'snapshot{number}_paired_predictions.npz',**saved)
            reports.append(report)
    result = dict(model_sha256=digest, reports=reports,
        scope='Original 550/575 intervention check. Actual/on overlaps training domain; uniform/on labels are new counterfactual solutions. No fitting or selection on original 600.',
        limitations='Sparse weighted totals/radial samples, not pixel-map validation. Passing the total screen alone does not authorize x8 use. Exchange-off is never predicted with the exchange-on surrogate.')
    (out/'uniform_surrogate_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
