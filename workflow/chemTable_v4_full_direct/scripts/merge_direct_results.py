#!/usr/bin/env python3
"""Merge primary and retry DESPOTIC chunks into one indexed result file."""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np

from workflow_config import config_path, load_config, write_provenance


ROOT = Path(__file__).resolve().parents[1]
CONFIG, CONFIG_PATH, CONFIG_SHA256, CONFIG_CANONICAL = load_config()
INPUT_PATH = config_path(CONFIG, "direct_input")
PRIMARY_DIR = config_path(CONFIG, "primary_chunk_dir")
RETRY_INPUT_PATH = config_path(CONFIG, "retry_input")
RETRY_DIR = config_path(CONFIG, "retry_chunk_dir")
OUTPUT_PATH = config_path(CONFIG, "merged_result")
SUMMARY_PATH = OUTPUT_PATH.with_suffix(".summary.json")

OUTPUT_NAMES = (
    "Tg_K", "Td_K", "X_H2", "X_CO", "X_13CO", "X_C18O",
    "L_CO_1_0", "L_13CO_1_0", "L_C18O_1_0",
    "tau_CO_1_0", "tau_13CO_1_0", "tau_C18O_1_0",
    "Tex_CO_1_0_K", "Tex_13CO_1_0_K", "Tex_C18O_1_0_K",
    "TB_CO_1_0_K_km_s", "TB_13CO_1_0_K_km_s", "TB_C18O_1_0_K_km_s",
)


def complete_chunks(directory):
    paths = sorted(directory.glob("direct_chunk_*.h5"))
    if not paths:
        raise RuntimeError(f"No direct chunks found in {directory}")
    for path in paths:
        with h5py.File(path, "r") as handle:
            if not bool(handle.attrs.get("complete", False)):
                raise RuntimeError(f"Incomplete chunk: {path}")
            chunk_config = handle.attrs.get("workflow_config_sha256")
            if chunk_config is not None and chunk_config != CONFIG_SHA256:
                raise RuntimeError(f"Configuration mismatch in chunk: {path}")
    return paths


def main():
    primary_paths = complete_chunks(PRIMARY_DIR)
    retry_paths = complete_chunks(RETRY_DIR)
    with h5py.File(INPUT_PATH, "r") as source:
        if not bool(source.attrs.get("complete", False)):
            raise RuntimeError(f"Incomplete direct input: {INPUT_PATH}")
        input_config = source.attrs.get("workflow_config_sha256")
        if input_config is not None and input_config != CONFIG_SHA256:
            raise RuntimeError("Direct input configuration does not match merge configuration")
        total = int(source.attrs["active_particle_count"])
        total_snapshot = int(source.attrs["total_particle_count"])
        particle_index = np.asarray(source["particle_index"][:], dtype=np.int64)
        oxygen_scale = float(source.attrs["oxygen_renorm_scale"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with h5py.File(temporary, "w") as target:
        target.attrs["complete"] = False
        write_provenance(
            target.attrs, CONFIG_PATH, CONFIG_SHA256, CONFIG_CANONICAL
        )
        target.attrs["input_path"] = str(INPUT_PATH)
        target.attrs["primary_chunk_dir"] = str(PRIMARY_DIR)
        target.attrs["retry_input_path"] = str(RETRY_INPUT_PATH)
        target.attrs["retry_chunk_dir"] = str(RETRY_DIR)
        target.attrs["active_particle_count"] = total
        target.attrs["total_snapshot_particle_count"] = total_snapshot
        target.attrs["oxygen_renorm_scale"] = oxygen_scale
        target.attrs["status_definition"] = "1=primary, 2=retry, -1=unconverged"
        target.create_dataset("particle_index", data=particle_index, compression="lzf")
        status_out = target.create_dataset(
            "status", shape=(total,), dtype=np.int8, chunks=(65536,),
            compression="lzf", fillvalue=-1,
        )
        output = {
            name: target.create_dataset(
                name, shape=(total,), dtype=np.float64, chunks=(65536,),
                compression="lzf", shuffle=True, fillvalue=np.nan,
            )
            for name in OUTPUT_NAMES
        }

        expected_start = 0
        primary_success = 0
        for path in primary_paths:
            with h5py.File(path, "r") as chunk:
                start = int(chunk.attrs["input_start"])
                stop = int(chunk.attrs["input_stop"])
                if start != expected_start:
                    raise RuntimeError(
                        f"Primary coverage gap: expected {expected_start}, got {start}"
                    )
                if not np.array_equal(
                    chunk["particle_index"][:], particle_index[start:stop]
                ):
                    raise RuntimeError(f"Particle-index mismatch in {path}")
                state = np.asarray(chunk["status"][:], dtype=np.int8)
                success = state == 1
                status_out[start:stop] = np.where(success, 1, -1)
                primary_success += int(np.count_nonzero(success))
                for name in OUTPUT_NAMES:
                    output[name][start:stop] = chunk[name][:]
                expected_start = stop
        if expected_start != total:
            raise RuntimeError(
                f"Primary chunks cover {expected_start} rows but input has {total}"
            )

        with h5py.File(RETRY_INPUT_PATH, "r") as retry_input:
            retry_config = retry_input.attrs.get("workflow_config_sha256")
            if retry_config is not None and retry_config != CONFIG_SHA256:
                raise RuntimeError("Retry input configuration does not match merge configuration")
            retry_total = int(retry_input.attrs["active_particle_count"])
            retry_source_rows = np.asarray(
                retry_input["source_input_row"][:], dtype=np.int64
            )
            retry_particle_index = np.asarray(
                retry_input["particle_index"][:], dtype=np.int64
            )
        if not np.array_equal(
            particle_index[retry_source_rows], retry_particle_index
        ):
            raise RuntimeError("Retry input does not match primary particle indices")

        expected_start = 0
        retry_success = 0
        for path in retry_paths:
            with h5py.File(path, "r") as chunk:
                start = int(chunk.attrs["input_start"])
                stop = int(chunk.attrs["input_stop"])
                if start != expected_start:
                    raise RuntimeError(
                        f"Retry coverage gap: expected {expected_start}, got {start}"
                    )
                source_rows = retry_source_rows[start:stop]
                if not np.array_equal(
                    chunk["particle_index"][:], particle_index[source_rows]
                ):
                    raise RuntimeError(f"Retry particle-index mismatch in {path}")
                state = np.asarray(chunk["status"][:], dtype=np.int8)
                success = state == 1
                successful_rows = source_rows[success]
                status_out[successful_rows] = 2
                retry_success += int(np.count_nonzero(success))
                for name in OUTPUT_NAMES:
                    values = np.asarray(chunk[name][:], dtype=float)
                    output[name][successful_rows] = values[success]
                expected_start = stop
        if expected_start != retry_total:
            raise RuntimeError(
                f"Retry chunks cover {expected_start} rows but input has {retry_total}"
            )

        final_status = np.asarray(status_out[:], dtype=np.int8)
        remaining_rows = np.flatnonzero(final_status < 0).astype(np.int64)
        target.create_dataset(
            "unconverged_input_row", data=remaining_rows, compression="lzf"
        )
        target.create_dataset(
            "unconverged_particle_index",
            data=particle_index[remaining_rows],
            compression="lzf",
        )
        target.attrs["primary_converged_count"] = primary_success
        target.attrs["retry_converged_count"] = retry_success
        target.attrs["final_converged_count"] = int(np.count_nonzero(final_status > 0))
        target.attrs["final_unconverged_count"] = remaining_rows.size
        target.attrs["final_convergence_fraction"] = float(
            np.count_nonzero(final_status > 0) / total
        )
        target.attrs["complete"] = True
        target.flush()
    os.replace(temporary, OUTPUT_PATH)

    summary = {
        "workflow_config_path": str(CONFIG_PATH),
        "workflow_config_sha256": CONFIG_SHA256,
        "input_path": str(INPUT_PATH),
        "output_path": str(OUTPUT_PATH),
        "active_particle_count": total,
        "total_snapshot_particle_count": total_snapshot,
        "primary_converged_count": primary_success,
        "retry_converged_count": retry_success,
        "final_converged_count": primary_success + retry_success,
        "final_unconverged_count": total - primary_success - retry_success,
        "final_convergence_fraction": (primary_success + retry_success) / total,
        "oxygen_renorm_scale": oxygen_scale,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
