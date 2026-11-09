"""Select intervention residual surrogates on prepartitioned original 550/575."""
import hashlib
import json
from pathlib import Path

import h5py
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from adaptive_local_surrogate import (AdaptiveLocalActualAbundanceSurrogate,
    OUTPUT_NAMES, raw_inputs_to_features, transform_outputs, inverse_transform_outputs)
from build_and_validate_adaptive_local_surrogate import target_weights

ROOT = Path(__file__).resolve().parent
ISO = ROOT.parent
OUT = ROOT/'intervention_models'
BRANCHES = ('actual_on','actual_off','uniform_on','uniform_off')


def load_data(base):
    parts = []
    audit = []
    for snapshot in (550,575):
        path = ROOT/f'output_calibration/snapshot{snapshot}_paired_retry.h5'
        with h5py.File(path) as f:
            assert bool(f.attrs['complete'])
            names = json.loads(f.attrs['output_names_json'])
            prov = json.loads(f.attrs['provenance_json'])
            assert prov['resolved_workflow']['snapshot']['index'] == snapshot
            ids = f['source_row'][:]
            assert len(np.unique(ids)) == len(ids)
            partition = f['validation_partition'][:]
            assert set(np.unique(partition)) == {0,1}
            good = np.logical_and.reduce([f[b+'/status'][:]==1 for b in BRANCHES])
            radius = f['radius_kpc'][:]
            mass = f['mass_Msun'][:]/f['inclusion_probability'][:]
            _, groups = np.unique(f['stratum'][:],axis=0,return_inverse=True)
            # Correct for the preassigned partition's within-stratum sampling too.
            w = mass.copy()
            for group in np.unique(groups):
                population = groups==group
                for part in (0,1):
                    mask = population & (partition==part)
                    if mask.any():
                        w[mask] *= population.sum()/mask.sum()
            negative = {}
            for branch in BRANCHES:
                raw = f['uniform_inputs' if branch.startswith('uniform') else 'actual_inputs'][:][good]
                vals = f[branch+'/values'][:][good]
                direct = vals[:,[names.index(n) for n in OUTPUT_NAMES]]
                assert np.all(np.isfinite(direct))
                negative[branch] = np.sum(direct<0,axis=0).tolist()
                direct = np.maximum(direct,0)
                predicted,_ = base.query(raw,chunk_size=10000)
                base_targets = transform_outputs(raw,np.column_stack([predicted[n] for n in OUTPUT_NAMES]))
                parts.append(dict(snapshot=snapshot,branch=branch,raw=raw,direct=direct,
                    features=np.column_stack((raw_inputs_to_features(raw),base_targets)),
                    baseline=base_targets,targets=transform_outputs(raw,direct),
                    radius=radius[good],weight=w[good],validation=partition[good]==1,ids=ids[good]))
            audit.append(dict(snapshot=snapshot,total=len(ids),common=int(good.sum()),
                train=int(np.sum(good & (partition==0))),validation=int(np.sum(good & (partition==1))),
                negative_values_clipped=negative,input_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return parts,audit


def evaluate(parts, predictions):
    records, scores = [], []
    totals = {}
    for data,pred in zip(parts,predictions):
        mask = data['validation']
        for lo,hi in [(0,float('inf'))]+[(i,i+1) for i in range(1,8)]:
            use = mask & (data['radius']>=lo) & (data['radius']<hi)
            if use.sum()<10:
                scores.append(float('inf'))
                continue
            a=np.sum(data['weight'][use,None]*data['direct'][use,3:],axis=0)
            b=np.sum(data['weight'][use,None]*pred[use,3:],axis=0)
            if np.any(a<=0) or np.any(b<=0):
                scores.append(float('inf'))
                continue
            bias=np.log10(b/a)
            ratio=float(bias[1]-bias[2])
            records.append(dict(snapshot=data['snapshot'],branch=data['branch'],r_lo=lo,
                r_hi=None if np.isinf(hi) else hi,count=int(use.sum()),
                line_bias_dex=bias.tolist(),ratio_bias_dex=ratio))
            # Global line/ratio and radial rare-ratio gates are preregistered.
            scores.extend(abs(bias).tolist() if np.isinf(hi) else [])
            scores.append(abs(ratio))
            totals[data['snapshot'],data['branch'],lo,hi]=ratio
    contrasts=[]
    for snapshot in sorted({d['snapshot'] for d in parts}):
        for lo,hi in [(0,float('inf'))]+[(i,i+1) for i in range(1,8)]:
            for a,b in [('actual_on','uniform_on'),('actual_off','uniform_off'),
                        ('actual_on','actual_off'),('uniform_on','uniform_off')]:
                ka,kb=(snapshot,a,lo,hi),(snapshot,b,lo,hi)
                if ka in totals and kb in totals:
                    error=totals[ka]-totals[kb]
                    scores.append(abs(error))
                    contrasts.append(dict(snapshot=snapshot,numerator=a,denominator=b,
                        r_lo=lo,r_hi=None if np.isinf(hi) else hi,contrast_bias_dex=error))
                else:
                    scores.append(float('inf'))
    worst=max(scores)
    return dict(max_gate_error_dex=worst,selection_screen_passed=bool(worst<=.05),
                branch_metrics=records,contrast_metrics=contrasts)


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'selection_report.json').exists():
        raise FileExistsError('Preserve prior model selection results')
    frozen=ISO/'chemTable_temporal_550_575/final_test_600'
    baseline_path=frozen/'frozen_surrogate.joblib'
    sha=hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    assert sha==json.loads((frozen/'frozen_manifest.json').read_text())['model_sha256']
    base=AdaptiveLocalActualAbundanceSurrogate(baseline_path,require_validated=False)
    parts,audit=load_data(base)
    candidates=[]
    # No correction is a diagnostic comparator, never an exchange-off physical model.
    reference=[inverse_transform_outputs(d['raw'],d['baseline'],base.bundle['target_clips']) for d in parts]
    reference_metrics=evaluate(parts,reference)
    for leaves in (15,31):
        models={}; clips={}; bounds={}
        for physics in ('on','off'):
            ds=[d for d in parts if d['branch'].endswith('_'+physics)]
            x=np.concatenate([d['features'][~d['validation']] for d in ds])
            target=np.concatenate([d['targets'][~d['validation']] for d in ds])
            baseline=np.concatenate([d['baseline'][~d['validation']] for d in ds])
            limits=np.array([[-40.,0.]]*3+[[-60.,float(target[:,j].max()+1)] for j in range(3,6)])
            clips[physics]=limits
            bounds[physics]=dict(min=x[:,:16].min(axis=0),max=x[:,:16].max(axis=0))
            models[physics]=[]
            for j in range(6):
                weights=[]
                for d in ds:
                    train=np.flatnonzero(~d['validation'])
                    weights.append(target_weights(j,train,d['weight'],d['direct'],d['radius'],
                                                  np.r_[np.arange(16),30.]))
                estimator=HistGradientBoostingRegressor(max_leaf_nodes=leaves,max_iter=400,
                    learning_rate=.05,min_samples_leaf=32,l2_regularization=1.,
                    early_stopping=False,random_state=20261010)
                estimator.fit(x,target[:,j]-baseline[:,j],sample_weight=np.concatenate(weights))
                models[physics].append(estimator)
                print(f'leaves={leaves} exchange={physics} target={j} fitted',flush=True)
        predictions=[]
        for d in parts:
            physics=d['branch'].rsplit('_',1)[1]
            residual=np.column_stack([m.predict(d['features']) for m in models[physics]])
            predictions.append(inverse_transform_outputs(d['raw'],d['baseline']+residual,clips[physics]))
        metrics=evaluate(parts,predictions)
        name=f'residual_leaves{leaves}'
        bundle=dict(models=models,target_clips=clips,feature_bounds=bounds,base_model=str(baseline_path),
            base_sha256=sha,output_names=OUTPUT_NAMES,feature_definition='16 existing physical features plus 6 frozen-baseline transformed outputs',
            selection_screen_passed=metrics['selection_screen_passed'],production_approved=False,
            method='separate nonlinear direct-label residual models for exchange on/off; no isotope-factor rescaling',
            input_audit=audit,hyperparameters=models['on'][0].get_params())
        joblib.dump(bundle,OUT/(name+'.joblib'),compress=3)
        np.savez_compressed(OUT/(name+'_validation_predictions.npz'),
            **{f'{d["snapshot"]}_{d["branch"]}_{key}':value[d['validation']]
               for d,p in zip(parts,predictions)
               for key,value in [('raw',d['raw']),('direct',d['direct']),('predicted',p),
                                 ('radius',d['radius']),('weight',d['weight']),('ids',d['ids'])]})
        (OUT/(name+'_metrics.json')).write_text(json.dumps(metrics,indent=2)+'\n')
        candidates.append(dict(name=name,score=metrics['max_gate_error_dex'],passed=metrics['selection_screen_passed']))
    best=min(candidates,key=lambda c:c['score'])
    report=dict(input_audit=audit,candidates=candidates,selected=best,
        baseline_as_off_is_not_physical=True,baseline_diagnostic=reference_metrics,
        production_approved=False,threshold_dex=.05,
        validation_scope='preassigned original550/575 validation rows, all versions grouped; model selection, not independent temporal evaluation',
        caveats='baseline already trained on actual550/575; intervention validation does not independently test that baseline. Failures are excluded, negative outputs clipped with counts. No 600 labels read; no x8 direct solves. No automatic promotion even if screening passes.',
        next_step='Freeze selected scheme, then independent direct counterfactual evaluation and projected closure plus x8 coverage. If screening fails, do not deploy.')
    (OUT/'selection_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(candidates=candidates,selected=best,production_approved=False)),flush=True)


if __name__=='__main__':
    main()
