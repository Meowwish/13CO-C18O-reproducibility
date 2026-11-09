"""Independent direct control test: never fits or selects using snapshot 600."""
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import joblib
import numpy as np
from adaptive_local_surrogate import (AdaptiveLocalActualAbundanceSurrogate, OUTPUT_NAMES,
    raw_inputs_to_features,transform_outputs,inverse_transform_outputs)
from train_intervention_models import evaluate,BRANCHES

ROOT=Path(__file__).resolve().parent


def main():
    cfg=json.loads(Path(os.environ['CONTROL_CONFIG']).read_text())
    assert cfg['snapshots']==[600]
    assert cfg['acceptance_limit_dex']==.05
    path=Path(cfg['frozen_model'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['frozen_model_sha256']
    bundle=joblib.load(path)
    assert bundle['selection_screen_passed'] and not bundle['production_approved']
    base_path=Path(bundle['base_model'])
    assert hashlib.sha256(base_path.read_bytes()).hexdigest()==bundle['base_sha256']
    if '--verify-only' in sys.argv:
        print('Frozen intervention and baseline model checksums verified',flush=True)
        return
    out=ROOT/cfg['output_subdir']
    source=out/'snapshot600_paired_retry.h5'
    base=AdaptiveLocalActualAbundanceSurrogate(base_path,require_validated=False)
    parts=[];predictions=[];coverage={}
    with h5py.File(source) as f:
        assert f.attrs['complete']
        prov=json.loads(f.attrs['provenance_json'])
        assert prov['resolved_workflow']['snapshot']['index']==600
        names=json.loads(f.attrs['output_names_json'])
        good=np.logical_and.reduce([f[b+'/status'][:]==1 for b in BRANCHES])
        w=f['mass_Msun'][:]/f['inclusion_probability'][:]
        radius=f['radius_kpc'][:]
        negative={}
        for branch in BRANCHES:
            raw=f['uniform_inputs' if branch.startswith('uniform') else 'actual_inputs'][:][good]
            direct=f[branch+'/values'][:][good][:,[names.index(n) for n in OUTPUT_NAMES]]
            assert np.all(np.isfinite(direct))
            negative[branch]=np.sum(direct<0,axis=0).tolist()
            direct=np.maximum(direct,0)
            initial,diag=base.query(raw)
            bt=transform_outputs(raw,np.column_stack([initial[n] for n in OUTPUT_NAMES]))
            features=raw_inputs_to_features(raw)
            aug=np.column_stack((features,bt))
            physics=branch.rsplit('_',1)[1]
            correction=np.column_stack([m.predict(aug) for m in bundle['models'][physics]])
            pred=inverse_transform_outputs(raw,bt+correction,bundle['target_clips'][physics])
            parts.append(dict(snapshot=600,branch=branch,raw=raw,direct=direct,
                radius=radius[good],weight=w[good],validation=np.ones(good.sum(),dtype=bool)))
            predictions.append(pred)
            bounds=bundle['feature_bounds'][physics]
            covered=np.all((features>=bounds['min']) & (features<=bounds['max']),axis=1)
            covered &= diag['within_axis_bounds'] & diag['within_joint_coverage']
            coverage[branch]=dict(weighted_mass_fraction=float(w[good][covered].sum()/w[good].sum()),
                direct_line_fractions=[float(np.sum(w[good][covered]*direct[covered,j])/np.sum(w[good]*direct[:,j]))
                                       for j in range(3,6)])
    metrics=evaluate(parts,predictions)
    result=dict(model_sha256=cfg['frozen_model_sha256'],input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        selected=len(good),common_converged=int(good.sum()),
        inverse_probability_mass_missing_fraction=float(w[~good].sum()/w.sum()),
        negative_labels_clipped=negative,metrics=metrics,coverage=coverage,
        production_approved=False,
        scope='Independent original600 direct counterfactual sample. Frozen before evaluation, no fitting or candidate selection.',
        limitations='Conditional on common convergence; particle-sampled radial sums are not full projected map validation. Coverage is not an error bound. x8 intervention coverage and maps remain separate work.')
    (out/'frozen_model_final_evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/'frozen_model_predictions.npz',
        **{d['branch']+'_'+k:v for d,p in zip(parts,predictions)
           for k,v in [('direct',d['direct']),('predicted',p),('raw',d['raw']),('radius',d['radius']),('weight',d['weight'])]})
    print(json.dumps(dict(common_converged=int(good.sum()),max_error_dex=metrics['max_gate_error_dex'],
        final_screen_passed=metrics['selection_screen_passed'],production_approved=False)),flush=True)


if __name__=='__main__':
    main()
