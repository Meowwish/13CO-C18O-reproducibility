"""Coverage-only transfer audit; reuse production input construction, no direct solves."""
import json
from pathlib import Path
import numpy as np
import process_snapshot563_lookup_v11 as pipeline

ROOT = Path(__file__).resolve().parent


def main():
    frozen = ROOT / "final_test_600"
    manifest = json.loads((frozen / "frozen_manifest.json").read_text())
    model = frozen / "frozen_surrogate.joblib"
    assert pipeline.file_sha256(model) == manifest["model_sha256"]
    summaries = []
    for snapshot in (563, 575):
        config_path = ROOT / "config" / f"x8_{snapshot}_coverage.json"
        settings = pipeline.load_settings(config_path)
        paths = pipeline.snapshot_files(settings)
        pipeline.validate_snapshot_header(paths, snapshot)
        context_path = pipeline.resolve_path(settings["outputs"]["context_npz"], config_path)
        if snapshot == 575:
            pipeline.prepare_context(config_path, settings)
        else:
            assert context_path.is_file()
        totals = {}
        for part in range(8):
            pipeline.process_part(config_path, settings, part)
            path = Path(settings["outputs"]["part_dir"]) / f"snapshot{snapshot}_part_{part:02d}_coverage.npz"
            with np.load(path) as data:
                assert str(data["surrogate_sha256"]) == manifest["model_sha256"]
                for key in ("source_particle_count", "active_particle_count", "active_mass_Msun",
                            "outside_coverage_count", "outside_coverage_mass_Msun",
                            "luminosity_sum_erg_s", "outside_coverage_luminosity_sum_erg_s"):
                    totals[key] = totals.get(key, 0) + data[key]
        assert totals["active_mass_Msun"] > 0 and np.all(totals["luminosity_sum_erg_s"] > 0)
        mass_fraction = totals["outside_coverage_mass_Msun"] / totals["active_mass_Msun"]
        light_fraction = totals["outside_coverage_luminosity_sum_erg_s"] / totals["luminosity_sum_erg_s"]
        record = {
            "snapshot": snapshot, "model_sha256": manifest["model_sha256"],
            "config_path": str(config_path),
            "scope": "Production neutral active gas within 30 kpc square face-on footprint; all eight snapshot pieces",
            "coverage_definitions": ["axis", "joint", "combined union of outside flags"],
            "line_order": ["CO", "13CO", "C18O"],
            "totals": {key: np.asarray(value).tolist() for key, value in totals.items()},
            "outside_mass_fractions": mass_fraction.tolist(),
            "outside_predicted_luminosity_fractions": light_fraction.tolist(),
            "passed": bool(mass_fraction[2] <= 0.01 and np.all(light_fraction[2, 1:] <= 0.01)),
            "limitation": "Model-input applicability only, not a direct x8 accuracy test; predicted luminosity weights may understate poorly predicted out-of-domain emission."
        }
        out = ROOT / "x8_coverage"
        out.mkdir(exist_ok=True)
        (out / f"snapshot{snapshot}_coverage.json").write_text(json.dumps(record, indent=2) + "\n")
        summaries.append(record)
    lines = ["# Frozen temporal model: x8 coverage", "",
             "Combined coverage requires both feature-axis and joint-domain membership.", "",
             "| Snapshot | Covered active mass [%] | Covered predicted CO [%] | Covered predicted 13CO [%] | Covered predicted C18O [%] | Gate |",
             "|---|---:|---:|---:|---:|---|"]
    for r in summaries:
        covered_mass = 100 * (1 - r["outside_mass_fractions"][2])
        light = 100 * (1 - np.array(r["outside_predicted_luminosity_fractions"][2]))
        lines.append(f"| {r['snapshot']} | {covered_mass:.5f} | {light[0]:.5f} | {light[1]:.5f} | {light[2]:.5f} | {r['passed']} |")
    lines += ["", "Gate: >=99% active mass and each predicted rare-line luminosity covered.",
              summaries[0]["limitation"]]
    (ROOT / "x8_coverage/report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
