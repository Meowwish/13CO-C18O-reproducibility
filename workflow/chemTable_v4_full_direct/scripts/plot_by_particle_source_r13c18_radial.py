#!/usr/bin/env python3
"""Compare full-direct R13/18 radial profiles for three yield selections."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "output" / "source_branches"
DESTINATION = OUTPUT_ROOT / "by_particle_R13_18_radial_source_branches.png"
DATA_PATH = OUTPUT_ROOT / "by_particle_R13_18_radial_source_branches.npz"
RADIAL_EDGES_KPC = np.arange(0.0, 16.0, 1.0)
NU_13CO_HZ = 1.102013543e11
NU_C18O_HZ = 1.097821734e11

CASES = (
    (
        "all",
        "All yields",
        ROOT / "output" / "snapshot600_full_direct" / "snapshot_600_maps_chuhan_125pc.npz",
        "#0072B2",
    ),
    (
        "slug_only",
        "SLUG particles: SNII + WR + young AGB",
        OUTPUT_ROOT / "slug_only" / "maps" / "snapshot_600_maps_chuhan_125pc.npz",
        "#D55E00",
    ),
    (
        "old_only",
        "Old stars: old AGB + SNIa + NSM",
        OUTPUT_ROOT / "old_only" / "maps" / "snapshot_600_maps_chuhan_125pc.npz",
        "#009E73",
    ),
)


def scalar(data, name):
    return np.asarray(data[name]).item()


def validate_case(data, expected_mode, path):
    mode = str(scalar(data, "stellar_yield_mode"))
    if mode != expected_mode:
        raise RuntimeError(f"{path} has stellar_yield_mode={mode}, expected {expected_mode}")
    if "direct_method" not in data.files or "per-particle" not in str(scalar(data, "direct_method")):
        raise RuntimeError(f"{path} is not a per-particle product")
    if "workflow_config_json" in data.files:
        config_json = str(scalar(data, "workflow_config_json"))
    elif "direct_input_path" in data.files:
        input_path = Path(str(scalar(data, "direct_input_path")))
        with h5py.File(input_path, "r") as source:
            config_json = str(source.attrs.get("workflow_config_json", ""))
    else:
        config_json = ""
    if not config_json:
        raise RuntimeError(f"{path} lacks full-direct solver provenance")
    config = json.loads(config_json)
    solver = config["chemistry"]["setChemEq"]
    if not np.isclose(float(solver["small_abundance"]), 1.0e-15):
        raise RuntimeError(f"{path} was not calculated with smallabd=1e-15")
    if solver["tEqGuess_s"] is not None:
        raise RuntimeError(f"{path} was not calculated with auto-tEqGuess")


def annular_ratio(map_l13, map_l18, pixel_size_pc):
    size_kpc = map_l13.shape[0] * pixel_size_pc / 1000.0
    coordinates = (np.arange(map_l13.shape[0]) + 0.5) * pixel_size_pc / 1000.0
    coordinates -= 0.5 * size_kpc
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="ij")
    radius = np.hypot(xx, yy)
    profile = np.full(RADIAL_EDGES_KPC.size - 1, np.nan)
    counts = np.zeros(profile.size, dtype=np.int64)
    frequency_factor = (NU_C18O_HZ / NU_13CO_HZ) ** 3
    finite = (
        np.isfinite(map_l13) & np.isfinite(map_l18)
        & (map_l13 >= 0.0) & (map_l18 > 0.0)
    )
    for index, (lower, upper) in enumerate(
        zip(RADIAL_EDGES_KPC[:-1], RADIAL_EDGES_KPC[1:])
    ):
        selected = finite & (radius >= lower) & (radius < upper)
        counts[index] = np.count_nonzero(selected)
        denominator = np.sum(map_l18[selected])
        if denominator > 0.0:
            profile[index] = np.sum(map_l13[selected]) / denominator * frequency_factor
    return profile, counts


def main():
    profiles = {}
    counts = {}
    for mode, _, path, _ in CASES:
        if not path.exists():
            raise FileNotFoundError(f"Required full-direct source product is missing: {path}")
        with np.load(path) as data:
            validate_case(data, mode, path)
            pixel_size_pc = float(scalar(data, "pixel_size_pc"))
            profiles[mode], counts[mode] = annular_ratio(
                np.asarray(data["mapL13CO"], dtype=float),
                np.asarray(data["mapLC18O"], dtype=float),
                pixel_size_pc,
            )

    radii = 0.5 * (RADIAL_EDGES_KPC[:-1] + RADIAL_EDGES_KPC[1:])
    figure, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    for mode, label, _, color in CASES:
        axis.plot(radii, profiles[mode], marker="o", linewidth=2.0, color=color, label=label)
    axis.set_xlabel("Galactocentric radius [kpc]")
    axis.set_ylabel(r"$R_{13/18}=W_{^{13}\mathrm{CO}(1-0)}/W_{\mathrm{C}^{18}\mathrm{O}(1-0)}$")
    axis.set_xlim(0.0, 15.0)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    axis.set_title(
        r"Snapshot 600: per-particle DESPOTIC, $\mathtt{smallabd}=10^{-15}$, "
        r"auto-$t_{\rm EqGuess}$" "\n125 pc beam; annular ratio of summed intensities"
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    figure.savefig(DESTINATION, dpi=220)
    plt.close(figure)
    np.savez_compressed(
        DATA_PATH,
        radii_kpc=radii,
        radial_edges_kpc=RADIAL_EDGES_KPC,
        ratio_all=profiles["all"],
        ratio_slug_only=profiles["slug_only"],
        ratio_old_only=profiles["old_only"],
        contributing_pixels_all=counts["all"],
        contributing_pixels_slug_only=counts["slug_only"],
        contributing_pixels_old_only=counts["old_only"],
        beam_fwhm_pc=125.0,
        profile_statistic="sum(W13) / sum(W18) in 1 kpc annuli",
    )
    print(f"Saved: {DESTINATION}")
    print(f"Saved: {DATA_PATH}")


if __name__ == "__main__":
    main()
