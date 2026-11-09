#!/usr/bin/env python3
"""Project merged direct-DESPOTIC particle results into snapshot-600 maps."""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from meshoid import Meshoid
import numpy as np

from workflow_config import config_path, load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG, CONFIG_PATH, CONFIG_SHA256, CONFIG_CANONICAL = load_config()
INPUT_PATH = config_path(CONFIG, "direct_input")
MERGED_PATH = config_path(CONFIG, "merged_result")
OUTPUT_DIR = config_path(CONFIG, "map_output_dir")
LEGACY_MAP_PATH = config_path(CONFIG, "legacy_map")
PROJECTION = CONFIG["projection"]
MAP_SIZE_PC = float(PROJECTION["map_size_pc"])
MAP_RESOLUTION = int(PROJECTION["map_resolution"])
XH = float(CONFIG["snapshot"]["hydrogen_mass_fraction"])
MSUN_G = 1.98847e33
MH_G = 1.6735575e-24
PC_CM = 3.0856775814913673e18
K_B = 1.380649e-16
C_LIGHT = 2.99792458e10
LINE_FREQUENCY_HZ = PROJECTION["line_frequencies_Hz"]


def rebin_mean(values, target_shape):
    yfactor = values.shape[0] // target_shape[0]
    xfactor = values.shape[1] // target_shape[1]
    if values.shape != (target_shape[0] * yfactor, target_shape[1] * xfactor):
        raise ValueError(f"Cannot rebin {values.shape} to {target_shape}")
    return values.reshape(target_shape[0], yfactor, target_shape[1], xfactor).mean(axis=(1, 3))


def project(coordinates, smoothing, weights):
    mesh = Meshoid(coordinates, np.asarray(weights, dtype=float), smoothing)
    return mesh.SurfaceDensity(
        mesh.m,
        size=MAP_SIZE_PC,
        res=MAP_RESOLUTION,
        center=np.zeros(3),
    )


def ratio(numerator, denominator):
    result = np.full_like(numerator, np.nan, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator > 0.0)
    return result


def to_w(map_luminosity, frequency_hz):
    intensity = map_luminosity / (4.0 * np.pi * PC_CM**2)
    return intensity * C_LIGHT**3 / (2.0 * K_B * frequency_hz**3 * 1.0e5)


def positive_summary(values):
    values = np.asarray(values, dtype=float)
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        return {"count": 0}
    return {
        "count": int(positive.size),
        "median": float(np.median(positive)),
        "p90": float(np.percentile(positive, 90.0)),
        "p99": float(np.percentile(positive, 99.0)),
        "max": float(np.max(positive)),
    }


def plot_six_panel(path, abundance_maps, intensity_maps):
    panels = [
        (r"$^{12}$CO abundance surface", abundance_maps["CO"], "viridis"),
        (r"$^{13}$CO abundance surface", abundance_maps["13CO"], "viridis"),
        (r"C$^{18}$O abundance surface", abundance_maps["C18O"], "viridis"),
        (r"$W_{12}$ [K km s$^{-1}$]", intensity_maps["CO"], "inferno"),
        (r"$W_{13}$ [K km s$^{-1}$]", intensity_maps["13CO"], "inferno"),
        (r"$W_{18}$ [K km s$^{-1}$]", intensity_maps["C18O"], "inferno"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.7), constrained_layout=True)
    for ax, (title, values, cmap) in zip(axes.flat, panels):
        positive = values[np.isfinite(values) & (values > 0.0)]
        if positive.size:
            vmax = float(np.percentile(positive, 99.5))
            vmin = max(float(np.min(positive)), vmax / 1.0e5)
            shown = np.where(values > 0.0, values, np.nan)
            image = ax.imshow(
                shown, origin="lower", extent=[-15, 15, -15, 15],
                cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax),
            )
        else:
            image = ax.imshow(values, origin="lower", extent=[-15, 15, -15, 15], cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")
        ax.set_aspect("equal")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with h5py.File(INPUT_PATH, "r") as source:
        coordinates = np.asarray(source["coordinates_pc"][:], dtype=float)
        smoothing = np.asarray(source["smoothing_length_pc"][:], dtype=float)
        masses = np.asarray(source["mass_Msun"][:], dtype=float)
        particle_index = np.asarray(source["particle_index"][:], dtype=np.int64)
        nh_map_native = np.asarray(source["NH_map_cm2"][:], dtype=float)
        oxygen_scale = float(source.attrs["oxygen_renorm_scale"])
        total_snapshot = int(source.attrs["total_particle_count"])
    with h5py.File(MERGED_PATH, "r") as result:
        if not bool(result.attrs.get("complete", False)):
            raise RuntimeError(f"Merged result is incomplete: {MERGED_PATH}")
        result_config = result.attrs.get("workflow_config_sha256")
        if result_config is not None and result_config != CONFIG_SHA256:
            raise RuntimeError("Merged result configuration does not match map configuration")
        if not np.array_equal(result["particle_index"][:], particle_index):
            raise RuntimeError("Merged result particle indices do not match input")
        status = np.asarray(result["status"][:], dtype=np.int8)
        fields = {
            name: np.asarray(result[name][:], dtype=float)
            for name in (
                "Tg_K", "X_H2", "X_CO", "X_13CO", "X_C18O",
                "L_CO_1_0", "L_13CO_1_0", "L_C18O_1_0",
                "tau_CO_1_0", "tau_13CO_1_0", "tau_C18O_1_0",
            )
        }
    converged = status > 0
    for values in fields.values():
        values[~converged] = 0.0

    abundance_maps = {
        "CO": project(coordinates, smoothing, masses * fields["X_CO"]),
        "13CO": project(coordinates, smoothing, masses * fields["X_13CO"]),
        "C18O": project(coordinates, smoothing, masses * fields["X_C18O"]),
    }
    hydrogen_nuclei = masses * MSUN_G * XH / MH_G
    particle_luminosity = {
        "CO": fields["L_CO_1_0"] * hydrogen_nuclei,
        "13CO": fields["L_13CO_1_0"] * hydrogen_nuclei,
        "C18O": fields["L_C18O_1_0"] * hydrogen_nuclei,
    }
    luminosity_maps = {
        name: project(coordinates, smoothing, values)
        for name, values in particle_luminosity.items()
    }
    intensity_maps = {
        name: to_w(luminosity_maps[name], LINE_FREQUENCY_HZ[name])
        for name in luminosity_maps
    }
    h2_co_map = (
        luminosity_maps["CO"]
        * float(PROJECTION["CO_inferred_H2_Msun_per_erg_s"])
    )
    nh_map = rebin_mean(nh_map_native, (MAP_RESOLUTION, MAP_RESOLUTION))
    map_yso = np.zeros_like(nh_map)
    if LEGACY_MAP_PATH.exists():
        with np.load(LEGACY_MAP_PATH) as legacy:
            if "mapYSO" in legacy.files and legacy["mapYSO"].shape == nh_map.shape:
                map_yso = np.asarray(legacy["mapYSO"], dtype=float)

    output_path = OUTPUT_DIR / "snapshot_600_maps_chuhan.npz"
    np.savez_compressed(
        output_path,
        NHmap=nh_map,
        CO_abundance=abundance_maps["CO"],
        C13O_abundance=abundance_maps["13CO"],
        C18O_abundance=abundance_maps["C18O"],
        H2_COMap=h2_co_map,
        mapLCO=luminosity_maps["CO"],
        mapL13CO=luminosity_maps["13CO"],
        mapLC18O=luminosity_maps["C18O"],
        mapR13=ratio(luminosity_maps["13CO"], luminosity_maps["CO"]),
        mapR18=ratio(luminosity_maps["C18O"], luminosity_maps["CO"]),
        mapR13C18=ratio(luminosity_maps["13CO"], luminosity_maps["C18O"]),
        mapYSO=map_yso,
        res=MAP_RESOLUTION,
        pixel_size_pc=MAP_SIZE_PC / MAP_RESOLUTION,
        stellar_yield_mode=CONFIG["snapshot"]["stellar_yield_mode"],
        include_old_stellar_yields=(
            CONFIG["snapshot"]["stellar_yield_mode"] in {"all", "old_only"}
        ),
        oxygen_renorm_enabled=CONFIG["isotope_fields"]["oxygen_renormalization"]["enabled"],
        oxygen_renorm_mode=CONFIG["isotope_fields"]["oxygen_renormalization"]["weighting"],
        oxygen_renorm_target_o16_c12=CONFIG["isotope_fields"]["oxygen_renormalization"]["target_16O_over_12C"],
        oxygen_renorm_scale_median=oxygen_scale,
        direct_method="per-particle GOW chemistry+thermal+LVG",
        direct_input_path=str(INPUT_PATH),
        direct_merged_path=str(MERGED_PATH),
        direct_active_particle_count=particle_index.size,
        direct_total_snapshot_particle_count=total_snapshot,
        direct_converged_count=np.count_nonzero(converged),
        direct_unconverged_count=np.count_nonzero(~converged),
        abundance_surface_map_definition="SurfaceDensity(Mgas * X_molecule)",
        workflow_config_path=str(CONFIG_PATH),
        workflow_config_sha256=CONFIG_SHA256,
        workflow_config_json=CONFIG_CANONICAL,
    )

    plot_six_panel(
        OUTPUT_DIR / "co_isotope_maps_snap600.png",
        abundance_maps,
        intensity_maps,
    )
    summary = {
        "workflow_config_path": str(CONFIG_PATH),
        "workflow_config_sha256": CONFIG_SHA256,
        "maps_path": str(output_path),
        "pixel_size_pc": MAP_SIZE_PC / MAP_RESOLUTION,
        "active_particle_count": int(particle_index.size),
        "converged_count": int(np.count_nonzero(converged)),
        "unconverged_count": int(np.count_nonzero(~converged)),
        "oxygen_renorm_scale": oxygen_scale,
        "mass_weighted_abundance_converged": {
            name: float(np.average(
                fields[f"X_{name}"][converged], weights=masses[converged]
            ))
            for name in ("H2", "CO", "13CO", "C18O")
        },
        "positive_pixel_W_K_km_s": {
            name: positive_summary(values) for name, values in intensity_maps.items()
        },
        "total_particle_luminosity_erg_s": {
            name: float(np.sum(values)) for name, values in particle_luminosity.items()
        },
        "particle_optical_depth": {
            "CO": positive_summary(fields["tau_CO_1_0"]),
            "13CO": positive_summary(fields["tau_13CO_1_0"]),
            "C18O": positive_summary(fields["tau_C18O_1_0"]),
        },
    }
    summary_path = OUTPUT_DIR / "direct_projection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
