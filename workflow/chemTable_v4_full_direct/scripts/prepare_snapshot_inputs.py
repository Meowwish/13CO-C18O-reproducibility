#!/usr/bin/env python3
"""Prepare per-particle snapshot inputs for full DESPOTIC/GOW solves."""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import yt
import yt.units as u
from meshoid import Meshoid

from workflow_config import config_path, load_config, write_provenance


ROOT = Path(__file__).resolve().parents[1]
CONFIG, CONFIG_PATH, CONFIG_SHA256, CONFIG_CANONICAL = load_config()
SNAPSHOT_PATH = config_path(CONFIG, "snapshot")
OUTPUT_PATH = config_path(CONFIG, "direct_input")
SUMMARY_PATH = OUTPUT_PATH.with_suffix(".summary.json")
FORCE = bool(CONFIG["execution"]["prepare"]["force"])
SNAPSHOT = CONFIG["snapshot"]
ISOTOPES = CONFIG["isotope_fields"]
OXYGEN = ISOTOPES["oxygen_renormalization"]
SHIELDING = CONFIG["shielding_column"]
XH = float(SNAPSHOT["hydrogen_mass_fraction"])
MAP_SIZE_PC = float(SHIELDING["map_size_pc"])
COLUMN_RESOLUTION = int(SHIELDING["map_resolution"])
MIN_DVDR_KM_S_PC = float(SHIELDING["minimum_dvdr_km_s_pc"])
SHIELDING_MODE = str(SHIELDING["mode"])
O16_C12_TARGET = float(OXYGEN["target_16O_over_12C"])
UNIT_BASE = SNAPSHOT["unit_base"]
BBOX = SNAPSHOT["bounding_box_code_units"]
PC_CM = 3.0856775814913673e18
GRAVITATIONAL_CONSTANT_CGS = 6.67430e-8


def raw_field(data, name):
    return np.asarray(data[("PartType0", name)].to_ndarray(), dtype=float)


def combined_mass_fraction(data, count, entries):
    result = np.zeros(count, dtype=float)
    for entry in entries:
        factor = entry.get("factor")
        if factor is None:
            factor = entry["numerator"] / entry["denominator"]
        result += raw_field(data, entry["field"]) * float(factor)
    return result


def isotope_number_abundances(data, masses):
    count = masses.size
    c12_mass = combined_mass_fraction(data, count, ISOTOPES["12C"])
    c13_mass = combined_mass_fraction(data, count, ISOTOPES["13C"])
    o16_mass = combined_mass_fraction(data, count, ISOTOPES["16O"])
    o18_mass = combined_mass_fraction(data, count, ISOTOPES["18O"])

    masses_atomic = ISOTOPES["atomic_mass_numbers"]
    x12 = c12_mass / (float(masses_atomic["12C"]) * XH)
    x13 = c13_mass / (float(masses_atomic["13C"]) * XH)
    x16_raw = o16_mass / (float(masses_atomic["16O"]) * XH)
    x18_raw = o18_mass / (float(masses_atomic["18O"]) * XH)
    valid = (
        np.isfinite(x12) & np.isfinite(x16_raw)
        & np.isfinite(masses) & (x12 > 0.0) & (x16_raw > 0.0)
        & (masses > 0.0)
    )
    if not np.any(valid):
        raise RuntimeError("No particles can be used for oxygen renormalisation")
    c12_mean = np.average(x12[valid], weights=masses[valid])
    o16_mean = np.average(x16_raw[valid], weights=masses[valid])
    oxygen_scale = (
        O16_C12_TARGET * c12_mean / o16_mean if OXYGEN["enabled"] else 1.0
    )
    return (
        x12,
        x13,
        x16_raw * oxygen_scale,
        x18_raw * (
            oxygen_scale if OXYGEN["apply_same_factor_to_18O"] else 1.0
        ),
        float(oxygen_scale),
        float(c12_mean),
        float(o16_mean),
    )


def finite_range(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"min": None, "median": None, "max": None}
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def write_dataset(handle, name, values, active):
    selected = np.asarray(values[active])
    chunks = True if selected.ndim == 1 else (min(65536, selected.shape[0]), selected.shape[1])
    handle.create_dataset(
        name,
        data=selected,
        chunks=chunks,
        compression="lzf",
        shuffle=True,
    )


def jeans_length_pc(density_g_cm3, internal_energy_code):
    gamma = float(SHIELDING["jeans_gamma"])
    unit_velocity = float(UNIT_BASE["UnitVelocity_in_cm_per_s"])
    sound_speed_squared = (
        gamma * (gamma - 1.0) * internal_energy_code * unit_velocity**2
    )
    valid = (
        np.isfinite(density_g_cm3) & (density_g_cm3 > 0.0)
        & np.isfinite(sound_speed_squared) & (sound_speed_squared > 0.0)
    )
    length = np.full(density_g_cm3.shape, np.nan, dtype=float)
    length[valid] = np.sqrt(
        np.pi * sound_speed_squared[valid]
        / (GRAVITATIONAL_CONSTANT_CGS * density_g_cm3[valid])
    ) / PC_CM
    return length


def local_sobolev_length_pc(mesh, density_g_cm3, smoothing_length_pc, jeans_pc):
    print("Computing the local density gradient for the Sobolev column")
    density_gradient = np.asarray(mesh.D(density_g_cm3), dtype=float)
    gradient_norm = np.linalg.norm(density_gradient, axis=1)
    raw_length = np.full(density_g_cm3.shape, np.inf, dtype=float)
    valid_gradient = np.isfinite(gradient_norm) & (gradient_norm > 0.0)
    raw_length[valid_gradient] = (
        density_g_cm3[valid_gradient] / gradient_norm[valid_gradient]
    )
    minimum_length = (
        float(SHIELDING["minimum_smoothing_length_fraction"])
        * smoothing_length_pc
    )
    resolved_length = np.maximum(raw_length, minimum_length)
    if bool(SHIELDING.get("cap_at_jeans_length", True)):
        resolved_length = np.minimum(resolved_length, jeans_pc)
    return resolved_length


def main():
    if OUTPUT_PATH.exists() and not FORCE:
        with h5py.File(OUTPUT_PATH, "r") as handle:
            if bool(handle.attrs.get("complete", False)):
                existing_config = handle.attrs.get("workflow_config_sha256")
                if existing_config is not None and existing_config != CONFIG_SHA256:
                    raise RuntimeError(
                        "Existing input was created with a different workflow config; "
                        "set execution.prepare.force=true only after choosing a new output path"
                    )
                print(f"Complete input already exists: {OUTPUT_PATH}")
                return

    print(f"Loading snapshot: {SNAPSHOT_PATH}")
    dataset = yt.load(str(SNAPSHOT_PATH), unit_base=UNIT_BASE, bounding_box=BBOX)
    data = dataset.all_data()

    coordinates = np.asarray(
        data[("PartType0", "Coordinates")].in_units("pc").d, dtype=float
    )
    smoothing_length = np.asarray(
        data[("PartType0", "SmoothingLength")].in_units("pc").d, dtype=float
    )
    masses = np.asarray(
        data[("PartType0", "Masses")].in_units("Msun").d, dtype=float
    )
    density = data[("PartType0", "Density")].in_units("g/cm**3")
    density_cgs = np.asarray(density.d, dtype=float)
    n_h = np.asarray((density * XH / u.mh).in_units("cm**-3").d, dtype=float)
    dvdr = np.asarray(
        (data[("PartType0", "NormVelocityGradient")] * u.km / u.s / u.kpc)
        .in_units("km/(s*pc)").d,
        dtype=float,
    )
    dvdr = np.maximum(dvdr, MIN_DVDR_KM_S_PC)

    h_ii = raw_field(data, "GrackleHII")
    internal_energy = raw_field(data, "InternalEnergy")
    h_ii = h_ii.copy()
    h_ii[internal_energy > float(SNAPSHOT["hot_gas_internal_energy_threshold"])] = XH

    print("Constructing face-on shielding column")
    mesh = Meshoid(coordinates, masses, smoothing_length)
    sigma = mesh.SurfaceDensity(
        mesh.m,
        size=MAP_SIZE_PC,
        res=COLUMN_RESOLUTION,
        center=np.zeros(3),
    )
    n_h_map = (
        sigma * u.Msun / u.pc**2 * XH / u.mh
    ).in_units("cm**-2").d
    pixel_pc = MAP_SIZE_PC / COLUMN_RESOLUTION
    xpix = np.floor(coordinates[:, 0] / pixel_pc).astype(np.int64) + COLUMN_RESOLUTION // 2
    ypix = np.floor(coordinates[:, 1] / pixel_pc).astype(np.int64) + COLUMN_RESOLUTION // 2
    footprint = (
        (xpix >= 0) & (xpix < COLUMN_RESOLUTION)
        & (ypix >= 0) & (ypix < COLUMN_RESOLUTION)
    )
    face_on_column = np.zeros(masses.size, dtype=float)
    face_on_column[footprint] = n_h_map[xpix[footprint], ypix[footprint]]

    if SHIELDING_MODE == "projected_face_on":
        n_col = face_on_column
        shielding_length_pc = np.divide(
            n_col, n_h * PC_CM,
            out=np.full_like(n_col, np.nan),
            where=np.isfinite(n_h) & (n_h > 0.0),
        )
    else:
        jeans_pc = jeans_length_pc(density_cgs, internal_energy)
        if SHIELDING_MODE == "jeans":
            shielding_length_pc = jeans_pc
        elif SHIELDING_MODE == "local_sobolev":
            shielding_length_pc = local_sobolev_length_pc(
                mesh, density_cgs, smoothing_length, jeans_pc
            )
        else:
            raise RuntimeError(f"Unsupported shielding mode: {SHIELDING_MODE}")
        n_col = n_h * shielding_length_pc * PC_CM
    n_col[~footprint] = 0.0

    print("Reading isotope fields and applying the fixed mass-weighted oxygen branch")
    x12, x13, x16, x18, oxygen_scale, c12_mean, o16_raw_mean = (
        isotope_number_abundances(data, masses)
    )
    active = (
        footprint & (h_ii <= float(SNAPSHOT["maximum_neutral_hii_fraction"]))
        & np.isfinite(n_h) & (n_h > 0.0)
        & np.isfinite(n_col) & (n_col > 0.0)
        & np.isfinite(dvdr) & (dvdr > 0.0)
        & np.isfinite(masses) & (masses > 0.0)
        & np.isfinite(smoothing_length) & (smoothing_length > 0.0)
        & np.isfinite(x12) & (x12 > 0.0)
        & np.isfinite(x16) & (x16 > 0.0)
        & np.isfinite(x13) & (x13 >= 0.0)
        & np.isfinite(x18) & (x18 >= 0.0)
    )
    active_index = np.flatnonzero(active).astype(np.int64)
    print(f"Active direct solves: {active_index.size}/{masses.size}")
    print(f"Oxygen renormalisation factor: {oxygen_scale:.10g}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with h5py.File(temporary, "w") as handle:
        handle.attrs["complete"] = False
        write_provenance(
            handle.attrs, CONFIG_PATH, CONFIG_SHA256, CONFIG_CANONICAL
        )
        handle.attrs["snapshot_path"] = str(SNAPSHOT_PATH)
        handle.attrs["snapshot_index"] = int(SNAPSHOT["index"])
        handle.attrs["total_particle_count"] = masses.size
        handle.attrs["active_particle_count"] = active_index.size
        handle.attrs["map_size_pc"] = MAP_SIZE_PC
        handle.attrs["column_resolution"] = COLUMN_RESOLUTION
        handle.attrs["minimum_dvdr_km_s_pc"] = MIN_DVDR_KM_S_PC
        handle.attrs["oxygen_renorm_target_o16_c12"] = O16_C12_TARGET
        handle.attrs["oxygen_renorm_scale"] = oxygen_scale
        handle.attrs["stellar_yield_mode"] = SNAPSHOT["stellar_yield_mode"]
        handle.attrs["shielding_mode"] = SHIELDING["mode"]
        handle.create_dataset("particle_index", data=active_index, compression="lzf")
        write_dataset(handle, "coordinates_pc", coordinates, active)
        write_dataset(handle, "smoothing_length_pc", smoothing_length, active)
        write_dataset(handle, "mass_Msun", masses, active)
        write_dataset(handle, "nH_cm3", n_h, active)
        write_dataset(handle, "NH_cm2", n_col, active)
        write_dataset(handle, "shielding_length_pc", shielding_length_pc, active)
        write_dataset(handle, "dVdr_km_s_pc", dvdr, active)
        write_dataset(handle, "x12C", x12, active)
        write_dataset(handle, "x13C", x13, active)
        write_dataset(handle, "x16O", x16, active)
        write_dataset(handle, "x18O", x18, active)
        handle.create_dataset("NH_map_cm2", data=n_h_map, compression="lzf", shuffle=True)
        handle.attrs["complete"] = True
        handle.flush()
    os.replace(temporary, OUTPUT_PATH)

    summary = {
        "workflow_config_path": str(CONFIG_PATH),
        "workflow_config_sha256": CONFIG_SHA256,
        "snapshot_path": str(SNAPSHOT_PATH),
        "output_path": str(OUTPUT_PATH),
        "total_particle_count": int(masses.size),
        "active_particle_count": int(active_index.size),
        "active_fraction": float(active_index.size / masses.size),
        "oxygen_renorm_target_o16_c12": O16_C12_TARGET,
        "oxygen_renorm_scale": oxygen_scale,
        "shielding_mode": SHIELDING_MODE,
        "mass_weighted_x12C": c12_mean,
        "mass_weighted_x16O_raw": o16_raw_mean,
        "ranges_active": {
            "nH_cm3": finite_range(n_h[active]),
            "NH_cm2": finite_range(n_col[active]),
            "shielding_length_pc": finite_range(shielding_length_pc[active]),
            "dVdr_km_s_pc": finite_range(dvdr[active]),
            "x12C": finite_range(x12[active]),
            "x13C": finite_range(x13[active]),
            "x16O": finite_range(x16[active]),
            "x18O": finite_range(x18[active]),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(f"Saved input: {OUTPUT_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
