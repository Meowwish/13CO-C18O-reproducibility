#!/usr/bin/env python3
"""Extract failed primary solves into a compact retry input file."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np

from workflow_config import config_path, load_config, write_provenance


ROOT = Path(__file__).resolve().parents[1]
CONFIG, CONFIG_PATH, CONFIG_SHA256, CONFIG_CANONICAL = load_config()
SOURCE_PATH = config_path(CONFIG, "direct_input")
PRIMARY_DIR = config_path(CONFIG, "primary_chunk_dir")
RETRY_PATH = config_path(CONFIG, "retry_input")
INPUT_NAMES = (
    "nH_cm3", "NH_cm2", "dVdr_km_s_pc", "x12C", "x13C", "x16O", "x18O",
)


def main():
    chunk_paths = sorted(PRIMARY_DIR.glob("direct_chunk_*.h5"))
    if not chunk_paths:
        raise RuntimeError(f"No primary chunks found in {PRIMARY_DIR}")

    failed_rows = []
    expected_start = 0
    for path in chunk_paths:
        with h5py.File(path, "r") as result:
            if not bool(result.attrs.get("complete", False)):
                raise RuntimeError(f"Incomplete primary chunk: {path}")
            chunk_config = result.attrs.get("workflow_config_sha256")
            if chunk_config is not None and chunk_config != CONFIG_SHA256:
                raise RuntimeError(f"Configuration mismatch in primary chunk: {path}")
            start = int(result.attrs["input_start"])
            stop = int(result.attrs["input_stop"])
            if start != expected_start:
                raise RuntimeError(
                    f"Primary chunk coverage gap: expected {expected_start}, got {start}"
                )
            status = np.asarray(result["status"][:], dtype=np.int8)
            failed_rows.append(start + np.flatnonzero(status != 1))
            expected_start = stop
    failed_rows = np.concatenate(failed_rows).astype(np.int64)

    with h5py.File(SOURCE_PATH, "r") as source:
        total = int(source.attrs["active_particle_count"])
        if expected_start != total:
            raise RuntimeError(
                f"Primary chunks cover {expected_start} rows but input contains {total}"
            )
        particle_index = np.asarray(source["particle_index"][:], dtype=np.int64)[failed_rows]
        values = {
            name: np.asarray(source[name][:], dtype=float)[failed_rows]
            for name in INPUT_NAMES
        }
        source_attrs = dict(source.attrs)

    RETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = RETRY_PATH.with_suffix(RETRY_PATH.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with h5py.File(temporary, "w") as target:
        for name, value in source_attrs.items():
            target.attrs[name] = value
        target.attrs["complete"] = False
        write_provenance(
            target.attrs, CONFIG_PATH, CONFIG_SHA256, CONFIG_CANONICAL
        )
        target.attrs["active_particle_count"] = failed_rows.size
        target.attrs["source_active_particle_count"] = total
        target.attrs["source_input_path"] = str(SOURCE_PATH)
        target.attrs["source_primary_chunk_dir"] = str(PRIMARY_DIR)
        target.attrs["selection"] = "primary status != 1"
        target.create_dataset("source_input_row", data=failed_rows, compression="lzf")
        target.create_dataset("particle_index", data=particle_index, compression="lzf")
        for name, value in values.items():
            target.create_dataset(name, data=value, compression="lzf", shuffle=True)
        target.attrs["complete"] = True
        target.flush()
    os.replace(temporary, RETRY_PATH)
    print(f"Primary input rows: {total}")
    print(f"Retry input rows: {failed_rows.size}")
    print(f"Saved: {RETRY_PATH}")


if __name__ == "__main__":
    main()
