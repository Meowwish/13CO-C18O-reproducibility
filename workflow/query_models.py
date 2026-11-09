"""Portable replay of the two frozen models on the probability sample."""
import hashlib
import json
from pathlib import Path
import sys

import h5py
import joblib
import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'chemTable_temporal_550_575/final_test_600'
sys.path.insert(0,str(BASE))
from adaptive_local_surrogate import (AdaptiveLocalActualAbundanceSurrogate,OUTPUT_NAMES,
    transform_outputs,raw_inputs_to_features,inverse_transform_outputs)


def main():
    path=BASE/'frozen_surrogate.joblib'
    model=AdaptiveLocalActualAbundanceSurrogate(path,require_validated=False)
    # The historical base_model path inside this bundle is deliberately not
    # followed: use the supplied relative model and check its exact identity.
    residual=joblib.load(ROOT/'isotope_fractionation_controls/intervention_models/residual_leaves15.joblib')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==residual['base_sha256']
    with h5py.File(ROOT/'x8_bounded_reference_20261101/sample.h5') as f:
        actual,slug,uniform=[f['sample/'+name][:] for name in ('actual','slug','uniform')]
    predictions=[]
    for raw in (actual,slug):
        values,_=model.query(raw,chunk_size=100000)
        predictions.append(np.column_stack([values[name] for name in OUTPUT_NAMES]))
    for raw in (actual,uniform):
        values,_=model.query(raw,chunk_size=100000)
        bt=transform_outputs(raw,np.column_stack([values[name] for name in OUTPUT_NAMES]))
        aug=np.column_stack((raw_inputs_to_features(raw),bt))
        for physics in ('on','off'):
            correction=np.column_stack([regressor.predict(aug) for regressor in residual['models'][physics]])
            predictions.append(inverse_transform_outputs(raw,bt+correction,residual['target_clips'][physics]))
    predictions=np.stack(predictions,axis=1)
    with np.load(ROOT/'x8_claim_checks_20261105_v3/predictions.npz') as f:previous=f['predicted']
    np.testing.assert_allclose(predictions,previous,rtol=2e-12,atol=1e-60)
    report=dict(passed=True,particles=len(actual),models=6,outputs=6,base_sha256=residual['base_sha256'],
        scope='Frozen prediction replay only; no retraining or physical validation.')
    (ROOT/'model_replay.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Portable frozen-model replay passed.',flush=True)


if __name__=='__main__':main()
