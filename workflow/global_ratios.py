"""Reconstruct published whole-frame ratios from cached full-frame integrals."""
import json
from pathlib import Path

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parent


def main():
    with h5py.File(ROOT/'x8_bounded_reference_20261101/sample.h5') as f:total=f['predicted_beam_totals'][0]
    ratio=total[:,1]/total[:,2]*(109782173400/110201354300)**3
    expected=np.array([8.268493532885843,6.488346623524451,8.257177152020528,6.492090793253663,7.736421080038711,5.994283669284608])
    np.testing.assert_allclose(ratio,expected,rtol=2e-12)
    report=dict(passed=True,ratios=ratio.tolist(),source_removal_percent=float(100*(ratio[1]/ratio[0]-1)),
        exchange_actual_percent=float(100*(ratio[2]/ratio[3]-1)),exchange_uniform_percent=float(100*(ratio[4]/ratio[5]-1)),
        scope='Reconstruction from cached complete-frame luminosity integrals, not re-projection of all galaxy particles.')
    (ROOT/'global_ratio_replay.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
