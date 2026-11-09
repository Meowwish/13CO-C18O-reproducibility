"""Evaluate the frozen temporal model using the established projection pipeline."""
import hashlib
import json
from pathlib import Path

import numpy as np
import validate_projected_closure as closure

ROOT = Path(__file__).resolve().parent
original_metric = closure.map_metric
original_scale = closure.evaluate_scale
original_report = closure.markdown_report


def metric(direct, lookup):
    result = original_metric(direct, lookup)
    positive = np.isfinite(direct) & (direct > 0)
    overlap = positive & np.isfinite(lookup) & (lookup > 0)
    result["direct_positive_lookup_nonpositive_count"] = int(np.sum(positive & ~overlap))
    if not np.any(positive):
        return result
    median, p90 = np.percentile(direct[positive], [50, 90])
    result["direct_positive_pixel_percentile_thresholds"] = [float(median), float(p90)]
    result["brightness_strata"] = {}
    for name, selected in {
        "faint_p0_p50": positive & (direct <= median),
        "intermediate_p50_p90": positive & (direct > median) & (direct <= p90),
        "bright_p90_p100": positive & (direct > p90),
    }.items():
        use = selected & overlap
        residual = np.log10(lookup[use] / direct[use])
        result["brightness_strata"][name] = {
            "direct_positive_count": int(selected.sum()),
            "positive_overlap_count": int(use.sum()),
            "unweighted_median_dex": float(np.median(residual)) if residual.size else None,
            "unweighted_p95_absolute_dex": float(np.percentile(abs(residual), 95)) if residual.size else None,
        }
    return result


def scale(direct, lookup):
    result = original_scale(direct, lookup)
    ratios = []
    for maps in (direct, lookup):
        a, b = maps["mapL13CO"], maps["mapLC18O"]
        valid = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        ratios.append(np.divide(a, b, out=np.full_like(a, np.nan), where=valid))
    valid = np.isfinite(ratios[0]) & np.isfinite(ratios[1])
    residual = np.log10(ratios[1][valid] / ratios[0][valid])
    result["ratio_map"] = {
        "positive_overlap_count": int(valid.sum()),
        "unweighted_median_dex": float(np.median(residual)),
        "unweighted_p95_absolute_dex": float(np.percentile(abs(residual), 95)),
        "note": "Energy-luminosity ratio; its lookup/direct residual equals that in brightness-temperature intensity ratio. No detection cut.",
    }
    return result


def report(summary):
    text = original_report(summary).replace("# v11 projected closure against direct DESPOTIC",
        "# Frozen 550+575 model: snapshot600 face-on test")
    text += "\nNo model fitting or selection uses this final test.\n"
    for name, record in summary["scales"].items():
        text += f"\n## {name}: brightness-stratified pixel errors\n\n"
        text += "Strata use unweighted percentiles of positive direct pixels; errors use positive overlap only.\n\n"
        text += "| Map | Stratum | Direct pixels | Overlap pixels | Median [dex] | P95 absolute [dex] |\n|---|---|---:|---:|---:|---:|\n"
        for field, values in record["maps"].items():
            for stratum, m in values.get("brightness_strata", {}).items():
                text += f"| {field} | {stratum} | {m['direct_positive_count']} | {m['positive_overlap_count']} | {m['unweighted_median_dex']} | {m['unweighted_p95_absolute_dex']} |\n"
        text += "\nRatio-map diagnostics: " + json.dumps(record["ratio_map"]) + "\n"
    return text


if __name__ == "__main__":
    frozen = ROOT / "final_test_600"
    manifest = json.loads((frozen / "frozen_manifest.json").read_text())
    assert hashlib.sha256((frozen / "frozen_surrogate.joblib").read_bytes()).hexdigest() == manifest["model_sha256"]
    closure.map_metric = metric
    closure.evaluate_scale = scale
    closure.markdown_report = report
    closure.main()
