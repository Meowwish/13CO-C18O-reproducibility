#!/usr/bin/env python3
"""Run the 125 pc and sensitivity plotting stages from the central config."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from workflow_config import config_path, load_config


ROOT = Path(__file__).resolve().parents[1]
ISOTOPE_ROOT = ROOT.parent
CONFIG, CONFIG_PATH, CONFIG_SHA256, _ = load_config()
SNAPSHOT = CONFIG["snapshot"]
MOCK = CONFIG["mock_observation"]
PROJECTION = CONFIG["projection"]
OUTPUT_DIR = config_path(CONFIG, "map_output_dir")


def run(command, environment):
    print("Running:", " ".join(str(item) for item in command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["OUTPUT_DIR_OVERRIDE"] = str(OUTPUT_DIR)
    environment["SNAPSHOT_INDICES"] = str(SNAPSHOT["index"])
    environment["TARGET_RESOLUTION_PC"] = str(MOCK["beam_pc"])
    environment["MAP_SIZE_PC"] = str(PROJECTION["map_size_pc"])
    environment["GAUSSIAN_TRUNCATE"] = str(MOCK["gaussian_truncate_sigma"])
    environment["NU_13CO_10_HZ"] = str(PROJECTION["line_frequencies_Hz"]["13CO"])
    environment["NU_C18O_10_HZ"] = str(PROJECTION["line_frequencies_Hz"]["C18O"])
    environment["HYDROGEN_MASS_FRACTION"] = str(
        CONFIG["snapshot"]["hydrogen_mass_fraction"]
    )
    environment["YOUNG_STAR_AGE_MYR"] = str(
        CONFIG["diagnostics"]["young_star_age_Myr"]
    )

    commands = []
    commands.append([
        sys.executable,
        str(ISOTOPE_ROOT / "chemTable_v2_visser_fractionation" / "scripts" / "postprocess_125pc.py"),
    ])
    for noise in MOCK["noise_levels_K_km_s"]:
        commands.append([
            sys.executable,
            str(ISOTOPE_ROOT / "chemTable_v2_visser_fractionation" / "scripts" / "plot_mock_r13c18_limits.py"),
            "--snapshot", str(SNAPSHOT["index"]),
            "--sigma13", str(noise["sigma13"]),
            "--sigma18", str(noise["sigma18"]),
            "--noise-frac", str(MOCK["noise_fraction_mode"]),
            "--hex-size-pc", str(MOCK["hex_size_pc"]),
            "--snr-threshold", str(MOCK["detection_snr_threshold"]),
        ])

    reference = MOCK["noise_levels_K_km_s"][0]
    snapshot_root = config_path(CONFIG, "snapshot").parents[1]
    commands.append([
        sys.executable,
        str(ISOTOPE_ROOT / "plot_r13c18_radial_noise_levels.py"),
        "--snapshot", str(SNAPSHOT["index"]),
        "--beam-pc", str(MOCK["beam_pc"]),
        "--hex-size-pc", str(MOCK["hex_size_pc"]),
        "--sigma13-mk", f"{1000.0 * float(reference['sigma13']):.12g}",
        "--sigma18-mk", f"{1000.0 * float(reference['sigma18']):.12g}",
        "--decades", str(MOCK["radial_sensitivity_decades"]),
        "--radial-bin-kpc", str(MOCK["radial_bin_kpc"]),
        "--max-radius-kpc", str(MOCK["maximum_radius_kpc"]),
        "--min-detections", str(MOCK["minimum_detections_per_radial_bin"]),
        "--snr-threshold", str(MOCK["detection_snr_threshold"]),
        "--snapshot-root", str(snapshot_root),
        "--output-dir", str(OUTPUT_DIR),
    ])

    for command in commands:
        if args.dry_run:
            print("Would run:", " ".join(str(item) for item in command))
        else:
            run(command, environment)

    if args.dry_run:
        return

    manifest = {
        "workflow_config_path": str(CONFIG_PATH),
        "workflow_config_sha256": CONFIG_SHA256,
        "commands": [[str(item) for item in command] for command in commands],
    }
    path = OUTPUT_DIR / "configured_mock_plot_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
