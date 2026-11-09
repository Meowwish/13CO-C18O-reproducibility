"""Build a reproducible 550+575 direct reference for temporal surrogate training."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT_NAMES = (
    "nH_cm3", "NH_cm2", "dVdr_km_s_pc", "x12C", "x13C", "x16O", "x18O",
    "mass_Msun", "coordinates_pc", "particle_index",
)
RESULT_NAMES = (
    "status", "particle_index", "X_CO", "X_13CO", "X_C18O",
    "L_CO_1_0", "L_13CO_1_0", "L_C18O_1_0",
)
LUMINOSITY_NAMES = ("L_CO_1_0", "L_13CO_1_0", "L_C18O_1_0")
CHUNK = 200_000


def source_paths(snapshot):
    branch = ROOT / f"snapshot{snapshot}"
    return (
        branch / "input" / f"snapshot_{snapshot}_direct_inputs.h5",
        branch / "output" / f"snapshot_{snapshot}_direct_merged.h5",
    )


def config(handle):
    raw = handle.attrs["workflow_config_json"]
    return json.loads(raw.decode("ascii") if isinstance(raw, bytes) else raw)


def source_check(snapshot, source, result):
    a, b = config(source), config(result)
    if a != b or int(a["snapshot"]["index"]) != snapshot:
        raise RuntimeError(f"Snapshot {snapshot}: input/result configuration mismatch")
    if not bool(result.attrs.get("complete", False)):
        raise RuntimeError(f"Snapshot {snapshot}: merged result incomplete")
    solver = a["chemistry"]["setChemEq"]
    if (solver["small_abundance"] != 1e-15 or solver["tEqGuess_s"] is not None
            or solver["relative_tolerance"] != 1e-3
            or a["cloud"]["cosmic_ray_ionization_rate_s-1"] != 1e-16):
        raise RuntimeError(f"Snapshot {snapshot}: unexpected solver settings")
    count = len(source["particle_index"])
    if any(len(source[name]) != count for name in INPUT_NAMES):
        raise RuntimeError(f"Snapshot {snapshot}: incomplete direct input")
    if any(len(result[name]) != count for name in RESULT_NAMES):
        raise RuntimeError(f"Snapshot {snapshot}: incomplete direct result")
    return a, count


def copy_field(output, name, source, count, offset, clip_nonpositive=False):
    target = output[name]
    for start in range(0, count, CHUNK):
        stop = min(start + CHUNK, count)
        values = source[name][start:stop]
        if clip_nonpositive:
            values = np.maximum(values, 0.0)
        target[offset + start:offset + stop] = values


def main():
    output = ROOT / "training_550_575"
    output.mkdir(exist_ok=True)
    paths = [source_paths(n) for n in (550, 575)]
    details = []
    with h5py.File(paths[0][0]) as first_input, h5py.File(paths[0][1]) as first_result:
        source_config, _ = source_check(550, first_input, first_result)
    for snapshot, (input_path, result_path) in zip((550, 575), paths):
        with h5py.File(input_path) as source, h5py.File(result_path) as result:
            current, count = source_check(snapshot, source, result)
            for key in ("chemistry", "cloud", "isotope_fields", "shielding_column"):
                if current[key] != source_config[key]:
                    raise RuntimeError(f"Snapshot {snapshot}: {key} differs from snapshot 550")
            details.append({"snapshot": snapshot, "count": count,
                            "input_path": str(input_path), "result_path": str(result_path)})
    total = sum(item["count"] for item in details)
    input_target = output / "direct_inputs_550_575.h5"
    result_target = output / "direct_results_550_575.h5"
    if input_target.exists() or result_target.exists():
        raise FileExistsError("Combined training files already exist")
    negative_counts = {str(n): {key: 0 for key in LUMINOSITY_NAMES} for n in (550, 575)}
    with h5py.File(input_target, "w") as combined_input, h5py.File(result_target, "w") as combined_result:
        with h5py.File(paths[0][0]) as example_input, h5py.File(paths[0][1]) as example_result:
            for names, example, target in ((INPUT_NAMES, example_input, combined_input),
                                            (RESULT_NAMES, example_result, combined_result)):
                for name in names:
                    shape = (total,) + example[name].shape[1:]
                    target.create_dataset(name, shape=shape, dtype=example[name].dtype,
                                          chunks=(min(CHUNK, total),) + shape[1:])
        offset = 0
        for item in details:
            n, count = item["snapshot"], item["count"]
            with h5py.File(item["input_path"]) as source, h5py.File(item["result_path"]) as result:
                for start in range(0, count, CHUNK):
                    stop = min(start + CHUNK, count)
                    if not np.array_equal(source["particle_index"][start:stop],
                                          result["particle_index"][start:stop]):
                        raise RuntimeError(f"Snapshot {n}: particle-index mismatch")
                for name in INPUT_NAMES:
                    copy_field(combined_input, name, source, count, offset)
                for name in RESULT_NAMES:
                    copy_field(combined_result, name, result, count, offset,
                               clip_nonpositive=name in LUMINOSITY_NAMES)
                for name in LUMINOSITY_NAMES:
                    for start in range(0, count, CHUNK):
                        stop = min(start + CHUNK, count)
                        negative_counts[str(n)][name] += int(np.count_nonzero(
                            (result["status"][start:stop] > 0)
                            & (result[name][start:stop] < 0)))
            offset += count
        combined_input.attrs["active_particle_count"] = total
        combined_result.attrs["complete"] = True
        combined_result.attrs["workflow_config_json"] = json.dumps(source_config)
        for target in (combined_input, combined_result):
            target.attrs["source_snapshots"] = np.asarray([550, 575])
            target.attrs["source_rows"] = np.asarray([item["count"] for item in details])
    manifest = {"sources": details, "negative_luminosities_set_to_zero": negative_counts,
                "train_snapshots": [550, 575], "excluded_final_test_snapshot": 600,
                "combined_input": str(input_target), "combined_result": str(result_target)}
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rng = np.random.default_rng(20261002)
    with h5py.File(output / "validation_rows_550_575.h5", "w") as rows:
        rows.create_dataset("validation/random_source_row",
                            data=np.sort(rng.choice(total, 200_000, replace=False)))
    template_path = ROOT.parent / "chemTable_v11_adaptive_local_actual_abundance/config/snapshot600_v11.json"
    settings = json.loads(template_path.read_text(encoding="ascii"))
    settings["run_name"] = "temporal_550_575_gradient_boosted_surrogate"
    settings["source"].update({
        "required_run_name": source_config["run_name"],
        "input_h5": str(input_target),
        "result_h5": str(result_target),
        "validation_rows_h5": str(output / "validation_rows_550_575.h5"),
        "source_manifest_json": str(output / "source_manifest.json"),
        "excluded_final_test_snapshot": 600,
    })
    settings["model"].update({
        "output_model": str(output / "temporal_550_575_surrogate.joblib"),
        "validation_json": str(output / "temporal_550_575_validation.json"),
        "validation_markdown": str(output / "temporal_550_575_validation.md"),
        "figure_dir": str(output / "figures"),
    })
    (output / "training_config.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
