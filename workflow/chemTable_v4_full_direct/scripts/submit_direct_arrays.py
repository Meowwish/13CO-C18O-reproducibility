#!/usr/bin/env python3
"""Submit Gadi-sized direct-solve arrays and record their job IDs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import json
import os
from pathlib import Path
import subprocess

import h5py

from workflow_config import config_path, load_config

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-job", required=True)
    parser.add_argument("--mode", choices=("primary", "retry"), default="primary")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--first-index", type=int, default=0)
    parser.add_argument("--last-index", type=int, default=None)
    parser.add_argument("--array-size", type=int, default=None)
    parser.add_argument("--pbs-script", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config, selected_config, config_sha256, _ = load_config(args.config)
    run = config["execution"][args.mode]
    input_path = config_path(
        config, "direct_input" if args.mode == "primary" else "retry_input"
    )
    if args.last_index is None:
        with h5py.File(input_path, "r") as source:
            row_count = int(source.attrs["active_particle_count"])
        args.last_index = math.ceil(row_count / int(run["chunk_size"])) - 1
    if args.array_size is None:
        args.array_size = int(config["execution"]["array_submission_max_subjobs"])
    if args.pbs_script is None:
        script_name = (
            "solve_snapshot600_array.pbs"
            if args.mode == "primary"
            else "retry_failed_snapshot600_array.pbs"
        )
        args.pbs_script = ROOT / "pbs" / script_name
    if args.manifest is None:
        manifest_name = (
            "direct_array_submissions.json"
            if args.mode == "primary"
            else "retry_array_submissions.json"
        )
        args.manifest = ROOT / "output" / manifest_name
    if args.first_index < 0 or args.last_index < args.first_index:
        raise ValueError("Invalid array index range")
    if not 1 <= args.array_size <= 10:
        raise ValueError("Gadi currently limits PBS arrays to at most 10 subjobs")
    if args.manifest.exists() and not args.dry_run:
        raise FileExistsError(
            f"Submission manifest already exists; refusing duplicate submission: {args.manifest}"
        )

    submissions = []
    for start in range(args.first_index, args.last_index + 1, args.array_size):
        stop = min(start + args.array_size - 1, args.last_index)
        variables = (
            f"CHEMTABLE_V4_CONFIG={selected_config},"
            f"DIRECT_RUN_MODE={args.mode}"
        )
        command = ["qsub", "-r", "y"]
        if start == stop:
            variables += f",DIRECT_CHUNK_INDEX={start}"
        else:
            command.extend(["-J", f"{start}-{stop}"])
        command.extend([
            "-W", f"depend=afterok:{args.prepare_job}",
            "-v", variables,
            str(args.pbs_script),
        ])
        if args.dry_run:
            print("Would run:", " ".join(command))
            continue
        process = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        record = {
            "start": start,
            "stop": stop,
            "returncode": process.returncode,
            "stdout": process.stdout.strip(),
            "stderr": process.stderr.strip(),
        }
        submissions.append(record)
        print(
            f"{start:04d}-{stop:04d}: returncode={process.returncode} "
            f"job={process.stdout.strip()} error={process.stderr.strip()}"
        )
        if process.returncode != 0:
            break

    if args.dry_run:
        return

    payload = {
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow_config_path": str(selected_config),
        "workflow_config_sha256": config_sha256,
        "mode": args.mode,
        "input_path": str(input_path),
        "prepare_job": args.prepare_job,
        "first_index": args.first_index,
        "last_index": args.last_index,
        "array_size": args.array_size,
        "pbs_script": str(args.pbs_script),
        "all_submissions_succeeded": bool(
            submissions
            and submissions[-1]["stop"] == args.last_index
            and all(item["returncode"] == 0 for item in submissions)
        ),
        "submissions": submissions,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    os.replace(temporary, args.manifest)
    if not payload["all_submissions_succeeded"]:
        raise RuntimeError(f"Array submission stopped early; see {args.manifest}")
    print(f"Submission manifest: {args.manifest}")


if __name__ == "__main__":
    main()
