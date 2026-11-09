"""Final temporal test of a frozen model; no fitting or DESPOTIC solves."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from adaptive_local_surrogate import AdaptiveLocalActualAbundanceSurrogate, RAW_INPUT_NAMES

ROOT = Path(__file__).resolve().parent
if ROOT.name == "final_test_600":
    ROOT = ROOT.parent
OUT = ROOT / "final_test_600"
LINES = ("L_CO_1_0", "L_13CO_1_0", "L_C18O_1_0")


def main():
    manifest = json.loads((OUT / "frozen_manifest.json").read_text())
    model_path = OUT / "frozen_surrogate.joblib"
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == manifest["model_sha256"]
    model = AdaptiveLocalActualAbundanceSurrogate(model_path, require_validated=False)
    base = ROOT.parent / "chemTable_v6_brightness_validation"
    input_path = base / "input/face_on_tight/snapshot_600_direct_inputs.h5"
    result_path = base / "output/face_on_tight/snapshot_600_direct_merged.h5"
    direct_sum = np.zeros(3)
    predicted_sum = np.zeros(3)
    mass_all = mass_used = 0.0
    selected_count = 0
    negative_counts = np.zeros(3, dtype=np.int64)
    with h5py.File(input_path) as src, h5py.File(result_path) as ref:
        config = json.loads(src.attrs["workflow_config_json"])
        assert config == json.loads(ref.attrs["workflow_config_json"])
        assert int(config["snapshot"]["index"]) == 600
        xh = float(config["snapshot"]["hydrogen_mass_fraction"])
        n = len(src["particle_index"])
        assert n == len(ref["particle_index"])
        for start in range(0, n, 100000):
            stop = min(n, start + 100000)
            sl = slice(start, stop)
            assert np.array_equal(src["particle_index"][sl], ref["particle_index"][sl])
            mass = np.asarray(src["mass_Msun"][sl], dtype=float)
            assert np.all(np.isfinite(mass) & (mass >= 0))
            use = ref["status"][sl] > 0
            mass_all += float(mass.sum())
            mass_used += float(mass[use].sum())
            selected_count += int(use.sum())
            if not np.any(use):
                continue
            raw = np.column_stack([src[name][sl][use] for name in RAW_INPUT_NAMES])
            prediction, _ = model.query(raw, chunk_size=100000)
            nuclei = mass[use] * 1.98847e33 * xh / 1.6735575e-24
            for j, name in enumerate(LINES):
                direct = np.asarray(ref[name][sl][use], dtype=float)
                pred = prediction[name]
                assert np.all(np.isfinite(direct)) and np.all(np.isfinite(pred))
                assert np.all(pred >= 0)
                negative_counts[j] += int(np.count_nonzero(direct < 0))
                direct_sum[j] += np.sum(np.maximum(direct, 0) * nuclei)
                predicted_sum[j] += np.sum(pred * nuclei)
            print(f"Evaluated {stop:,}/{n:,} rows", flush=True)
    assert np.all(direct_sum > 0) and np.all(predicted_sum > 0)
    bias = np.log10(predicted_sum / direct_sum)
    ratio_bias = float(bias[1] - bias[2])
    summary = {
        "frozen_model": manifest,
        "input_path": str(input_path), "reference_path": str(result_path),
        "scope": "All converged selected snapshot600 particles; no additional radial or detection cut",
        "luminosity_unit": "erg s^-1", "conversion": "sum(L_per_H * mass_Msun * 1.98847e33 * XH / 1.6735575e-24)",
        "input_count": n, "converged_count": selected_count,
        "excluded_mass_fraction": 1 - mass_used / mass_all,
        "small_negative_reference_luminosities_clipped_to_zero": dict(zip(LINES, negative_counts.tolist())),
        "lines": {name: {"direct": float(direct_sum[j]), "surrogate": float(predicted_sum[j]),
                         "bias_dex": float(bias[j]), "relative_percent": float(100 * (10**bias[j] - 1))}
                  for j, name in enumerate(LINES)},
        "ratio_L13_over_L18": {"direct": float(direct_sum[1] / direct_sum[2]),
                                "surrogate": float(predicted_sum[1] / predicted_sum[2]), "bias_dex": ratio_bias},
        "validation": {"total_line_bias_limit_dex": 0.05,
                       "total_lines_passed": bool(np.all(np.abs(bias) <= 0.05)),
                       "total_ratio_passed_0p05dex": abs(ratio_bias) <= 0.05},
        "caveat": "Integrated accuracy alone does not validate spatial or particle-level predictions. L13/L18 is an energy-luminosity ratio, not a K km/s intensity ratio; its surrogate/direct dex bias is unchanged by fixed line-unit conversions."
    }
    (OUT / "snapshot600_total_luminosity_test.json").write_text(json.dumps(summary, indent=2) + "\n")
    rows = ["# Frozen 550+575 model: final snapshot600 test", "",
            "Matched converged particles; total energy luminosities in erg/s; no fitting on snapshot600.", "",
            "| Line | Direct [erg/s] | Surrogate [erg/s] | Bias [dex] | Difference [%] |",
            "|---|---:|---:|---:|---:|"]
    for name, record in summary["lines"].items():
        rows.append(f"| {name} | {record['direct']:.6e} | {record['surrogate']:.6e} | {record['bias_dex']:+.5f} | {record['relative_percent']:+.3f} |")
    rows += ["", f"Total rare-line ratio bias: {ratio_bias:+.5f} dex.",
             f"Total-line gate passed: {summary['validation']['total_lines_passed']}.",
             "", summary["caveat"]]
    (OUT / "snapshot600_total_luminosity_test.md").write_text("\n".join(rows) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
