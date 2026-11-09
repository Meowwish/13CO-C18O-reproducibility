#!/usr/bin/env python3
"""Stream x8 snapshot 563 through the projection-validated v11 surrogate."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import time

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from meshoid import Meshoid
import numpy as np

from adaptive_local_surrogate import AdaptiveLocalActualAbundanceSurrogate
from workflow_config import load_config


MSUN_G = 1.98847e33
MH_G = 1.6735575e-24
PC_CM = 3.0856775814913673e18
K_B = 1.380649e-16
C_LIGHT = 2.99792458e10
MAP_FIELDS = (
    "abundance_CO",
    "abundance_13CO",
    "abundance_C18O",
    "luminosity_CO",
    "luminosity_13CO",
    "luminosity_C18O",
)
COVERAGE_NAMES = ("axis", "joint", "combined")


def resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def load_settings(config_path: Path) -> dict:
    settings = json.loads(config_path.read_text(encoding="ascii"))
    if settings.get("schema_version") != 1:
        raise ValueError("Only schema_version=1 is supported")
    return settings


def snapshot_files(settings: dict) -> list[Path]:
    paths = [Path(item).resolve() for item in glob.glob(settings["snapshot"]["file_pattern"])]

    def part_number(path: Path) -> int:
        match = re.search(r"\.(\d+)\.hdf5$", path.name)
        if match is None:
            raise ValueError(f"Cannot parse snapshot part number: {path}")
        return int(match.group(1))

    paths.sort(key=part_number)
    expected = int(settings["snapshot"]["expected_file_count"])
    if len(paths) != expected:
        raise RuntimeError(f"Expected {expected} snapshot files, found {len(paths)}")
    if [part_number(path) for path in paths] != list(range(expected)):
        raise RuntimeError("Snapshot part numbers are not contiguous from zero")
    return paths


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def validate_projected_closure(
    config_path: Path,
    settings: dict,
    model_path: Path,
) -> tuple[Path, str]:
    chemistry = settings["chemistry"]
    report_path = resolve_path(chemistry["projected_closure_report"], config_path)
    report = json.loads(report_path.read_text(encoding="ascii"))
    if report.get("run_name") != chemistry.get("required_closure_run_name", chemistry["required_surrogate_run_name"]):
        raise RuntimeError("Projected-closure run name does not match the configuration")
    if Path(report["model_path"]).resolve() != model_path:
        raise RuntimeError("Projected-closure report refers to a different surrogate")
    if report.get("direct_particle_source") != "original-resolution snapshot 600 only":
        raise RuntimeError("Projected closure was not derived from the approved direct run")
    if int(report.get("high_resolution_direct_rows", -1)) != 0:
        raise RuntimeError("High-resolution DESPOTIC rows are forbidden in this workflow")

    failures = set(report["validation"]["failures"])
    allowed = set(chemistry["allowed_full_report_failures"])
    if not failures.issubset(allowed):
        unexpected = sorted(failures - allowed)
        raise RuntimeError(f"Unexpected projected-closure failures: {unexpected}")

    p95_limit = float(chemistry["projected_map_p95_limit_dex"])
    integrated_limit = float(chemistry["projected_integrated_bias_limit_dex"])
    radial_limit = float(chemistry["projected_radial_bias_limit_dex"])
    minimum_radius = float(chemistry["minimum_validated_radius_kpc"])
    maximum_radius = float(chemistry["maximum_validated_radius_kpc"])
    for scale_name, scale in report["scales"].items():
        for map_name, metric in scale["maps"].items():
            if metric["direct_weighted_p95_absolute_error_dex"] > p95_limit:
                raise RuntimeError(f"{scale_name}/{map_name} exceeds the map p95 gate")
            if abs(metric["integrated_bias_dex"]) > integrated_limit:
                raise RuntimeError(f"{scale_name}/{map_name} exceeds the integrated gate")
        radial = scale["radial"]
        edges = np.asarray(radial["edges_kpc"], dtype=float)
        use = (edges[:-1] >= minimum_radius) & (edges[1:] <= maximum_radius)
        if not np.any(use):
            raise RuntimeError("Projected-closure report has no bins in the validated range")
        for line_name, values in radial["line_bias_dex"].items():
            if np.max(np.abs(np.asarray(values, dtype=float)[use])) > radial_limit:
                raise RuntimeError(
                    f"{scale_name}/{line_name} exceeds the radial gate outside the centre"
                )
        ratio_bias = np.asarray(
            radial["ratio_13CO_over_C18O_bias_dex"], dtype=float
        )
        if np.max(np.abs(ratio_bias[use])) > radial_limit:
            raise RuntimeError(
                f"{scale_name}/13CO-C18O ratio exceeds the radial gate outside the centre"
            )
    return report_path, file_sha256(report_path)


def atomic_write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def isotope_column(entry: dict) -> int:
    match = re.fullmatch(r"Metallicity_(\d+)", str(entry["field"]))
    if match is None:
        raise ValueError(f"Unsupported isotope field: {entry['field']}")
    return int(match.group(1))


def isotope_plan(source_config: dict) -> tuple[list[int], dict[str, list[tuple[int, float]]]]:
    definitions = source_config["isotope_fields"]
    plan: dict[str, list[tuple[int, float]]] = {}
    columns: set[int] = set()
    for isotope in ("12C", "13C", "16O", "18O"):
        records = []
        for entry in definitions[isotope]:
            column = isotope_column(entry)
            factor = entry.get("factor")
            if factor is None:
                factor = float(entry["numerator"]) / float(entry["denominator"])
            records.append((column, float(factor)))
            columns.add(column)
        plan[isotope] = records
    return sorted(columns), plan


def isotope_abundances(
    metallicity: np.ndarray,
    columns: list[int],
    plan: dict[str, list[tuple[int, float]]],
    hydrogen_mass_fraction: float,
) -> dict[str, np.ndarray]:
    local_column = {column: index for index, column in enumerate(columns)}
    mass_numbers = {"12C": 12.0, "13C": 13.0, "16O": 16.0, "18O": 18.0}
    result = {}
    for isotope, records in plan.items():
        mass_fraction = np.zeros(metallicity.shape[0], dtype=float)
        for column, factor in records:
            mass_fraction += metallicity[:, local_column[column]] * factor
        result[isotope] = mass_fraction / (
            mass_numbers[isotope] * hydrogen_mass_fraction
        )
    return result


def unit_factors(settings: dict) -> dict[str, float]:
    snapshot = settings["snapshot"]
    length_cm = float(snapshot["unit_length_cm"])
    mass_g = float(snapshot["unit_mass_g"])
    return {
        "length_pc": length_cm / PC_CM,
        "mass_Msun": mass_g / MSUN_G,
        "density_cgs": mass_g / length_cm**3,
        "dvdr_km_s_pc": (
            float(snapshot["unit_velocity_cm_s"]) / 1.0e5
        ) / (length_cm / PC_CM),
    }


def project_surface(
    coordinates_pc: np.ndarray,
    smoothing_pc: np.ndarray,
    weights: np.ndarray,
    map_size_pc: float,
    resolution: int,
) -> np.ndarray:
    if coordinates_pc.size == 0:
        return np.zeros((resolution, resolution), dtype=float)
    half_size = 0.5 * map_size_pc
    intersects = (
        (np.abs(coordinates_pc[:, 0]) <= half_size + smoothing_pc)
        & (np.abs(coordinates_pc[:, 1]) <= half_size + smoothing_pc)
        & np.isfinite(weights)
    )
    if not np.any(intersects):
        return np.zeros((resolution, resolution), dtype=float)
    mesh = Meshoid(
        coordinates_pc[intersects],
        np.asarray(weights[intersects], dtype=float),
        smoothing_pc[intersects],
        n_jobs=-1,
    )
    return mesh.SurfaceDensity(
        mesh.m,
        size=map_size_pc,
        res=resolution,
        center=np.zeros(3),
    )


def validate_snapshot_header(paths: list[Path], snapshot_index: int) -> int:
    total = 0
    expected_time = None
    for path in paths:
        with h5py.File(path, "r") as handle:
            header = handle["Header"].attrs
            if int(header["NumFilesPerSnapshot"]) != len(paths):
                raise RuntimeError(f"Header file count mismatch in {path}")
            local = int(np.asarray(header["NumPart_ThisFile"], dtype=np.int64)[0])
            if local != int(handle["PartType0/Masses"].shape[0]):
                raise RuntimeError(f"Gas-particle count mismatch in {path}")
            current_time = float(header["Time"])
            if expected_time is None:
                expected_time = current_time
            elif not np.isclose(current_time, expected_time):
                raise RuntimeError("Snapshot parts have inconsistent times")
            total += local
    if total <= 0:
        raise RuntimeError(f"Snapshot {snapshot_index} contains no gas particles")
    return total


def prepare_context(config_path: Path, settings: dict) -> None:
    started = time.perf_counter()
    paths = snapshot_files(settings)
    model_path = resolve_path(settings["chemistry"]["surrogate_model"], config_path)
    model = AdaptiveLocalActualAbundanceSurrogate(
        model_path, require_validated=False
    )
    validate_model(model, settings)
    validate_projected_closure(config_path, settings, model_path)
    snapshot = settings["snapshot"]
    projection = settings["projection"]
    output = resolve_path(settings["outputs"]["context_npz"], config_path)
    source_config, _, source_sha, source_canonical = load_config(
        settings["chemistry"]["source_workflow_config"]
    )
    expected_zeta = float(settings["chemistry"]["cosmic_ray_ionization_rate_s-1"])
    actual_zeta = float(source_config["cloud"]["cosmic_ray_ionization_rate_s-1"])
    if not np.isclose(actual_zeta, expected_zeta):
        raise RuntimeError(f"CR ionisation mismatch: {actual_zeta} != {expected_zeta}")
    columns, plan = isotope_plan(source_config)
    factors = unit_factors(settings)
    xh = float(snapshot["hydrogen_mass_fraction"])
    map_size = float(projection["map_size_pc"])
    resolution = int(projection["column_resolution"])
    chunk_size = int(projection["particle_chunk_size"])
    sigma_map = np.zeros((resolution, resolution), dtype=float)
    weighted_x12 = 0.0
    weighted_x16 = 0.0
    valid_mass = 0.0
    total_mass = 0.0
    particle_count = validate_snapshot_header(paths, int(snapshot["index"]))

    for file_index, path in enumerate(paths):
        print(f"Context pass {file_index + 1}/{len(paths)}: {path}", flush=True)
        with h5py.File(path, "r") as handle:
            group = handle["PartType0"]
            count = group["Masses"].shape[0]
            for start in range(0, count, chunk_size):
                stop = min(start + chunk_size, count)
                mass = np.asarray(group["Masses"][start:stop], dtype=float) * factors["mass_Msun"]
                coordinates = (
                    np.asarray(group["Coordinates"][start:stop], dtype=float)
                    * factors["length_pc"]
                )
                smoothing = (
                    np.asarray(group["SmoothingLength"][start:stop], dtype=float)
                    * factors["length_pc"]
                )
                metallicity = np.asarray(
                    group["Metallicity"][start:stop, columns], dtype=float
                )
                isotope = isotope_abundances(metallicity, columns, plan, xh)
                valid = (
                    np.isfinite(mass) & (mass > 0.0)
                    & np.isfinite(isotope["12C"]) & (isotope["12C"] > 0.0)
                    & np.isfinite(isotope["16O"]) & (isotope["16O"] > 0.0)
                )
                total_mass += float(np.sum(mass[np.isfinite(mass) & (mass > 0.0)]))
                valid_mass += float(np.sum(mass[valid]))
                weighted_x12 += float(np.sum(mass[valid] * isotope["12C"][valid]))
                weighted_x16 += float(np.sum(mass[valid] * isotope["16O"][valid]))
                sigma_map += project_surface(
                    coordinates, smoothing, mass, map_size, resolution
                )

    x12_mean = weighted_x12 / valid_mass
    x16_raw_mean = weighted_x16 / valid_mass
    oxygen = source_config["isotope_fields"]["oxygen_renormalization"]
    oxygen_scale = (
        float(oxygen["target_16O_over_12C"]) * x12_mean / x16_raw_mean
        if bool(oxygen["enabled"]) else 1.0
    )
    nh_map = sigma_map * MSUN_G / PC_CM**2 * xh / MH_G
    atomic_write_npz(
        output,
        NH_map_cm2=nh_map,
        oxygen_scale=np.asarray(oxygen_scale),
        mass_weighted_x12C=np.asarray(x12_mean),
        mass_weighted_x16O_raw=np.asarray(x16_raw_mean),
        total_gas_mass_Msun=np.asarray(total_mass),
        total_gas_particle_count=np.asarray(particle_count, dtype=np.int64),
        source_workflow_config_sha256=np.asarray(source_sha),
        source_workflow_config_json=np.asarray(source_canonical),
        snapshot_files=np.asarray([str(path) for path in paths]),
        elapsed_seconds=np.asarray(time.perf_counter() - started),
    )
    print(f"Saved global context: {output}", flush=True)
    print(f"Oxygen renormalisation factor: {oxygen_scale:.10g}", flush=True)


def validate_model(
    model: AdaptiveLocalActualAbundanceSurrogate,
    settings: dict,
) -> None:
    bundle_config = model.bundle.get("config", {})
    required = settings["chemistry"]["required_source_run_name"]
    actual = bundle_config.get("source", {}).get("required_run_name")
    if actual != required:
        raise RuntimeError(f"Surrogate source run mismatch: {actual!r} != {required!r}")
    required_surrogate = settings["chemistry"]["required_surrogate_run_name"]
    if bundle_config.get("run_name") != required_surrogate:
        raise RuntimeError("Surrogate run name does not match the configuration")
    source_zeta = float(
        load_config(settings["chemistry"]["source_workflow_config"])[0]["cloud"][
            "cosmic_ray_ionization_rate_s-1"
        ]
    )
    expected = float(settings["chemistry"]["cosmic_ray_ionization_rate_s-1"])
    if not np.isclose(source_zeta, expected):
        raise RuntimeError(f"Surrogate source zeta mismatch: {source_zeta} != {expected}")


def process_part(config_path: Path, settings: dict, part_index: int) -> None:
    started = time.perf_counter()
    paths = snapshot_files(settings)
    if not 0 <= part_index < len(paths):
        raise ValueError(f"part-index must be in [0,{len(paths) - 1}]")
    path = paths[part_index]
    context_path = resolve_path(settings["outputs"]["context_npz"], config_path)
    output_dir = resolve_path(settings["outputs"]["part_dir"], config_path)
    coverage_only = bool(settings.get("coverage_only", False))
    suffix = "coverage" if coverage_only else "maps"
    output_path = output_dir / f"snapshot{settings['snapshot']['index']}_part_{part_index:02d}_{suffix}.npz"
    model_path = resolve_path(settings["chemistry"]["surrogate_model"], config_path)
    model = AdaptiveLocalActualAbundanceSurrogate(
        model_path, require_validated=False
    )
    validate_model(model, settings)
    closure_path, closure_sha = validate_projected_closure(
        config_path, settings, model_path
    )
    model_sha = file_sha256(model_path)
    source_config = load_config(settings["chemistry"]["source_workflow_config"])[0]
    columns, plan = isotope_plan(source_config)
    with np.load(context_path) as context:
        nh_map = np.asarray(context["NH_map_cm2"], dtype=float)
        oxygen_scale = float(context["oxygen_scale"])

    snapshot = settings["snapshot"]
    projection = settings["projection"]
    factors = unit_factors(settings)
    xh = float(snapshot["hydrogen_mass_fraction"])
    map_size = float(projection["map_size_pc"])
    column_resolution = int(projection["column_resolution"])
    map_resolution = int(projection["map_resolution"])
    chunk_size = int(projection["particle_chunk_size"])
    column_pixel_pc = map_size / column_resolution
    minimum_dvdr = float(projection["minimum_dvdr_km_s_pc"])
    maps = {} if coverage_only else {name: np.zeros((map_resolution, map_resolution), dtype=float) for name in MAP_FIELDS}
    active_count = 0
    active_mass = 0.0
    outside_count = np.zeros(len(COVERAGE_NAMES), dtype=np.int64)
    outside_mass = np.zeros(len(COVERAGE_NAMES))
    abundance_mass_sum = np.zeros(3)
    luminosity_sum = np.zeros(3)
    outside_luminosity_sum = np.zeros((len(COVERAGE_NAMES), 3))
    maximum_coverage_distance2 = 0.0

    print(f"Processing snapshot part {part_index}: {path}", flush=True)
    with h5py.File(path, "r") as handle:
        group = handle["PartType0"]
        source_count = group["Masses"].shape[0]
        for start in range(0, source_count, chunk_size):
            stop = min(start + chunk_size, source_count)
            coordinates = (
                np.asarray(group["Coordinates"][start:stop], dtype=float)
                * factors["length_pc"]
            )
            smoothing = (
                np.asarray(group["SmoothingLength"][start:stop], dtype=float)
                * factors["length_pc"]
            )
            mass = np.asarray(group["Masses"][start:stop], dtype=float) * factors["mass_Msun"]
            density = np.asarray(group["Density"][start:stop], dtype=float) * factors["density_cgs"]
            n_h = density * xh / MH_G
            dvdr = np.maximum(
                np.asarray(group["NormVelocityGradient"][start:stop], dtype=float)
                * factors["dvdr_km_s_pc"],
                minimum_dvdr,
            )
            h_ii = np.asarray(group["GrackleHII"][start:stop], dtype=float)
            internal_energy = np.asarray(group["InternalEnergy"][start:stop], dtype=float)
            h_ii[internal_energy > float(snapshot["hot_gas_internal_energy_threshold"])] = xh
            metallicity = np.asarray(group["Metallicity"][start:stop, columns], dtype=float)
            isotope = isotope_abundances(metallicity, columns, plan, xh)
            x12 = isotope["12C"]
            x13 = isotope["13C"]
            x16 = isotope["16O"] * oxygen_scale
            x18 = isotope["18O"] * (
                oxygen_scale
                if source_config["isotope_fields"]["oxygen_renormalization"][
                    "apply_same_factor_to_18O"
                ] else 1.0
            )
            xpix = np.floor(coordinates[:, 0] / column_pixel_pc).astype(np.int64) + column_resolution // 2
            ypix = np.floor(coordinates[:, 1] / column_pixel_pc).astype(np.int64) + column_resolution // 2
            footprint = (
                (xpix >= 0) & (xpix < column_resolution)
                & (ypix >= 0) & (ypix < column_resolution)
            )
            n_col = np.zeros(stop - start, dtype=float)
            n_col[footprint] = nh_map[xpix[footprint], ypix[footprint]]
            active = (
                footprint
                & (h_ii <= float(snapshot["maximum_neutral_hii_fraction"]))
                & np.isfinite(n_h) & (n_h > 0.0)
                & np.isfinite(n_col) & (n_col > 0.0)
                & np.isfinite(dvdr) & (dvdr > 0.0)
                & np.isfinite(mass) & (mass > 0.0)
                & np.isfinite(smoothing) & (smoothing > 0.0)
                & np.isfinite(x12) & (x12 > 0.0)
                & np.isfinite(x13) & (x13 >= 0.0)
                & np.isfinite(x16) & (x16 > 0.0)
                & np.isfinite(x18) & (x18 >= 0.0)
            )
            if not np.any(active):
                continue
            local_raw = np.column_stack((
                n_h[active], n_col[active], dvdr[active],
                x12[active], x13[active], x16[active], x18[active],
            ))
            outputs, diagnostic = model.query(local_raw, chunk_size=chunk_size)
            physical = np.column_stack([
                outputs[name] for name in (
                    "X_CO", "X_13CO", "X_C18O",
                    "L_CO_1_0", "L_13CO_1_0", "L_C18O_1_0",
                )
            ])
            if np.any(~np.isfinite(physical)) or np.any(physical < 0.0):
                raise RuntimeError(f"Non-physical surrogate output in part {part_index}")
            local_mass = mass[active]
            hydrogen_nuclei = local_mass * MSUN_G * xh / MH_G
            weight_matrix = np.column_stack((
                local_mass[:, None] * physical[:, :3],
                hydrogen_nuclei[:, None] * physical[:, 3:],
            ))
            local_coordinates = coordinates[active]
            local_smoothing = smoothing[active]
            for column, name in enumerate(() if coverage_only else MAP_FIELDS):
                maps[name] += project_surface(
                    local_coordinates,
                    local_smoothing,
                    weight_matrix[:, column],
                    map_size,
                    map_resolution,
                )
            within_axis = np.asarray(diagnostic["within_axis_bounds"], dtype=bool)
            within_joint = np.asarray(diagnostic["within_joint_coverage"], dtype=bool)
            outside_masks = (
                ~within_axis,
                ~within_joint,
                ~(within_axis & within_joint),
            )
            active_count += int(local_mass.size)
            active_mass += float(np.sum(local_mass))
            abundance_mass_sum += np.sum(local_mass[:, None] * physical[:, :3], axis=0)
            luminosity_sum += np.sum(hydrogen_nuclei[:, None] * physical[:, 3:], axis=0)
            for coverage_index, outside in enumerate(outside_masks):
                outside_count[coverage_index] += int(np.count_nonzero(outside))
                outside_mass[coverage_index] += float(np.sum(local_mass[outside]))
                outside_luminosity_sum[coverage_index] += np.sum(
                    hydrogen_nuclei[outside, None] * physical[outside, 3:], axis=0
                )
            distance2 = np.asarray(diagnostic["coverage_distance2"], dtype=float)
            finite_distance = distance2[np.isfinite(distance2)]
            if finite_distance.size:
                maximum_coverage_distance2 = max(
                    maximum_coverage_distance2, float(np.max(finite_distance))
                )
            print(
                f"part={part_index} rows={stop:,}/{source_count:,} "
                f"active={active_count:,} outside_combined={outside_count[2]:,}",
                flush=True,
            )

    atomic_write_npz(
        output_path,
        **maps,
        source_file=np.asarray(str(path)),
        source_particle_count=np.asarray(source_count, dtype=np.int64),
        active_particle_count=np.asarray(active_count, dtype=np.int64),
        active_mass_Msun=np.asarray(active_mass),
        outside_coverage_count=np.asarray(outside_count),
        outside_coverage_mass_Msun=np.asarray(outside_mass),
        abundance_mass_sum=np.asarray(abundance_mass_sum),
        luminosity_sum_erg_s=np.asarray(luminosity_sum),
        outside_coverage_luminosity_sum_erg_s=np.asarray(outside_luminosity_sum),
        maximum_coverage_distance2=np.asarray(maximum_coverage_distance2),
        surrogate_sha256=np.asarray(model_sha),
        projected_closure_report=np.asarray(str(closure_path)),
        projected_closure_sha256=np.asarray(closure_sha),
        elapsed_seconds=np.asarray(time.perf_counter() - started),
    )
    print(f"Saved part map: {output_path}", flush=True)


def rebin_mean(values: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    yfactor = values.shape[0] // target_shape[0]
    xfactor = values.shape[1] // target_shape[1]
    if values.shape != (target_shape[0] * yfactor, target_shape[1] * xfactor):
        raise ValueError(f"Cannot rebin {values.shape} to {target_shape}")
    return values.reshape(
        target_shape[0], yfactor, target_shape[1], xfactor
    ).mean(axis=(1, 3))


def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full_like(numerator, np.nan, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator > 0.0)
    return result


def luminosity_to_w(map_luminosity: np.ndarray, frequency_hz: float) -> np.ndarray:
    intensity = map_luminosity / (4.0 * np.pi * PC_CM**2)
    return intensity * C_LIGHT**3 / (2.0 * K_B * frequency_hz**3 * 1.0e5)


def positive_summary(values: np.ndarray) -> dict:
    positive = np.asarray(values, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size == 0:
        return {"count": 0}
    return {
        "count": int(positive.size),
        "median": float(np.median(positive)),
        "p90": float(np.percentile(positive, 90.0)),
        "p99": float(np.percentile(positive, 99.0)),
        "p99p9": float(np.percentile(positive, 99.9)),
        "max": float(np.max(positive)),
    }


def log_limits(values: np.ndarray, lower_percentile: float = 50.0) -> tuple[float, float]:
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        return 1.0, 10.0
    low, high = np.percentile(positive, [lower_percentile, 99.9])
    if not high > low:
        high = low * 10.0
    return float(max(low, np.nextafter(0.0, 1.0))), float(high)


def plot_maps(path: Path, abundance: dict, intensity: dict, snapshot_index: int) -> None:
    panels = [
        (r"$^{12}$CO abundance surface", abundance["CO"], "viridis"),
        (r"$^{13}$CO abundance surface", abundance["13CO"], "viridis"),
        (r"C$^{18}$O abundance surface", abundance["C18O"], "viridis"),
        (r"$W_{12}$ [K km s$^{-1}$]", intensity["CO"], "inferno"),
        (r"$W_{13}$ [K km s$^{-1}$]", intensity["13CO"], "inferno"),
        (r"$W_{18}$ [K km s$^{-1}$]", intensity["C18O"], "inferno"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.7), constrained_layout=True)
    for axis, (title, values, cmap) in zip(axes.flat, panels):
        vmin, vmax = log_limits(values)
        image = axis.imshow(
            np.where(values > 0.0, values, np.nan),
            origin="lower",
            extent=[-15, 15, -15, 15],
            cmap=cmap,
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
        axis.set_title(title)
        axis.set_xlabel("x [kpc]")
        axis.set_ylabel("y [kpc]")
        axis.set_aspect("equal")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(f"Snapshot {snapshot_index}: v11 actual-abundance surrogate", fontsize=14)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_co(path: Path, h2_co: np.ndarray, w_co: np.ndarray, snapshot_index: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.1), constrained_layout=True)
    panels = (
        (r"CO-inferred $\Sigma_{\rm H_2}$ [M$_\odot$ pc$^{-2}$]", h2_co, "viridis"),
        (r"$^{12}$CO(1-0) $W$ [K km s$^{-1}$]", w_co, "inferno"),
    )
    for axis, (title, values, cmap) in zip(axes, panels):
        vmin, vmax = log_limits(values)
        image = axis.imshow(
            np.where(values > 0.0, values, np.nan),
            origin="lower",
            extent=[-15, 15, -15, 15],
            cmap=cmap,
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
        axis.set_title(title)
        axis.set_xlabel("x [kpc]")
        axis.set_ylabel("y [kpc]")
        axis.set_aspect("equal")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(f"Snapshot {snapshot_index}: 20 pc pixels", fontsize=14)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def merge_parts(config_path: Path, settings: dict) -> None:
    started = time.perf_counter()
    output_paths = settings["outputs"]
    part_dir = resolve_path(output_paths["part_dir"], config_path)
    expected = int(settings["snapshot"]["expected_file_count"])
    parts = [part_dir / f"snapshot{settings['snapshot']['index']}_part_{index:02d}_maps.npz" for index in range(expected)]
    missing = [str(path) for path in parts if not path.exists()]
    if missing:
        raise RuntimeError("Missing part maps: " + ", ".join(missing))
    model_path = resolve_path(settings["chemistry"]["surrogate_model"], config_path)
    model_sha = file_sha256(model_path)
    closure_path, closure_sha = validate_projected_closure(
        config_path, settings, model_path
    )
    resolution = int(settings["projection"]["map_resolution"])
    maps = {name: np.zeros((resolution, resolution), dtype=float) for name in MAP_FIELDS}
    scalar_keys = (
        "source_particle_count", "active_particle_count", "active_mass_Msun",
    )
    totals = {key: 0.0 for key in scalar_keys}
    abundance_mass_sum = np.zeros(3)
    luminosity_sum = np.zeros(3)
    outside_count = np.zeros(len(COVERAGE_NAMES), dtype=np.int64)
    outside_mass = np.zeros(len(COVERAGE_NAMES))
    outside_luminosity_sum = np.zeros((len(COVERAGE_NAMES), 3))
    maximum_coverage_distance2 = 0.0
    for path in parts:
        with np.load(path) as data:
            if str(data["surrogate_sha256"].item()) != model_sha:
                raise RuntimeError(f"Surrogate hash mismatch in {path}")
            for name in MAP_FIELDS:
                maps[name] += np.asarray(data[name], dtype=float)
            for key in scalar_keys:
                totals[key] += float(data[key])
            abundance_mass_sum += np.asarray(data["abundance_mass_sum"], dtype=float)
            luminosity_sum += np.asarray(data["luminosity_sum_erg_s"], dtype=float)
            outside_count += np.asarray(data["outside_coverage_count"], dtype=np.int64)
            outside_mass += np.asarray(data["outside_coverage_mass_Msun"], dtype=float)
            outside_luminosity_sum += np.asarray(
                data["outside_coverage_luminosity_sum_erg_s"], dtype=float
            )
            maximum_coverage_distance2 = max(
                maximum_coverage_distance2,
                float(data["maximum_coverage_distance2"]),
            )
            if str(data["projected_closure_sha256"].item()) != closure_sha:
                raise RuntimeError(f"Projected-closure hash mismatch in {path}")

    context_path = resolve_path(output_paths["context_npz"], config_path)
    with np.load(context_path) as context:
        nh_map = rebin_mean(
            np.asarray(context["NH_map_cm2"], dtype=float),
            (resolution, resolution),
        )
        oxygen_scale = float(context["oxygen_scale"])
        total_gas_mass = float(context["total_gas_mass_Msun"])
        total_gas_count = int(context["total_gas_particle_count"])
    frequencies = settings["projection"]["line_frequencies_Hz"]
    abundance = {
        "CO": maps["abundance_CO"],
        "13CO": maps["abundance_13CO"],
        "C18O": maps["abundance_C18O"],
    }
    luminosity = {
        "CO": maps["luminosity_CO"],
        "13CO": maps["luminosity_13CO"],
        "C18O": maps["luminosity_C18O"],
    }
    intensity = {
        name: luminosity_to_w(values, float(frequencies[name]))
        for name, values in luminosity.items()
    }
    h2_co = luminosity["CO"] * float(
        settings["projection"]["CO_inferred_H2_Msun_per_erg_s"]
    )
    model = AdaptiveLocalActualAbundanceSurrogate(
        model_path, require_validated=False
    )
    validate_model(model, settings)
    particle_validation_passed = bool(model.validation_passed)
    active_mass = totals["active_mass_Msun"]
    coverage_mass_fraction = outside_mass / max(active_mass, 1.0e-300)
    coverage_luminosity_fraction = outside_luminosity_sum / np.maximum(
        luminosity_sum[None, :], 1.0e-300
    )
    maximum_uncovered_mass = float(
        settings["chemistry"]["maximum_uncovered_active_mass_fraction"]
    )
    minimum_reliable_rare = float(
        settings["chemistry"]["minimum_reliable_rare_line_luminosity_fraction"]
    )
    highres_transfer_gate_passed = bool(
        coverage_mass_fraction[2] <= maximum_uncovered_mass
        and np.all(1.0 - coverage_luminosity_fraction[2, 1:] >= minimum_reliable_rare)
    )
    config_canonical = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    maps_path = resolve_path(output_paths["maps_npz"], config_path)
    atomic_write_npz(
        maps_path,
        NHmap=nh_map,
        CO_abundance=abundance["CO"],
        C13O_abundance=abundance["13CO"],
        C18O_abundance=abundance["C18O"],
        H2_COMap=h2_co,
        mapLCO=luminosity["CO"],
        mapL13CO=luminosity["13CO"],
        mapLC18O=luminosity["C18O"],
        mapR13=ratio(intensity["13CO"], intensity["CO"]),
        mapR18=ratio(intensity["C18O"], intensity["CO"]),
        mapR13C18=ratio(intensity["13CO"], intensity["C18O"]),
        mapYSO=np.zeros_like(nh_map),
        res=np.asarray(resolution),
        pixel_size_pc=np.asarray(float(settings["projection"]["map_size_pc"]) / resolution),
        stellar_yield_mode=np.asarray(settings.get("stellar_yield_mode", "all")),
        include_old_stellar_yields=np.asarray(settings.get("stellar_yield_mode", "all") != "slug_only"),
        oxygen_renorm_enabled=np.asarray(True),
        oxygen_renorm_mode=np.asarray("gas_mass"),
        oxygen_renorm_target_o16_c12=np.asarray(2.0),
        oxygen_renorm_scale_median=np.asarray(oxygen_scale),
        direct_method=np.asarray(
            settings["chemistry"]["required_surrogate_run_name"] + "; no high-resolution DESPOTIC calls"
        ),
        direct_input_path=np.asarray(settings["snapshot"]["file_pattern"]),
        direct_merged_path=np.asarray(str(model_path)),
        direct_active_particle_count=np.asarray(int(totals["active_particle_count"])),
        direct_total_snapshot_particle_count=np.asarray(total_gas_count),
        direct_converged_count=np.asarray(0),
        direct_unconverged_count=np.asarray(0),
        lookup_active_particle_count=np.asarray(int(totals["active_particle_count"])),
        workflow_config_path=np.asarray(str(config_path)),
        workflow_config_sha256=np.asarray(hashlib.sha256(config_canonical.encode("ascii")).hexdigest()),
        workflow_config_json=np.asarray(config_canonical),
        lookup_particle_validation_passed=np.asarray(particle_validation_passed),
        lookup_projected_scope_validation_passed=np.asarray(True),
        lookup_highres_transfer_gate_passed=np.asarray(highres_transfer_gate_passed),
        lookup_quantitative_scope=np.asarray(settings["chemistry"]["quantitative_scope"]),
        lookup_minimum_validated_radius_kpc=np.asarray(
            float(settings["chemistry"]["minimum_validated_radius_kpc"])
        ),
        lookup_maximum_validated_radius_kpc=np.asarray(
            float(settings["chemistry"]["maximum_validated_radius_kpc"])
        ),
        lookup_projected_closure_report=np.asarray(str(closure_path)),
        lookup_projected_closure_sha256=np.asarray(closure_sha),
        cosmic_ray_ionization_rate_s_1=np.asarray(
            float(settings["chemistry"]["cosmic_ray_ionization_rate_s-1"])
        ),
        surrogate_sha256=np.asarray(model_sha),
    )

    summary = {
        "schema_version": 1,
        "run_name": settings["run_name"],
        "snapshot_index": int(settings["snapshot"]["index"]),
        "snapshot_file_count": expected,
        "total_gas_particle_count": total_gas_count,
        "source_particle_count_from_parts": int(totals["source_particle_count"]),
        "active_particle_count": int(totals["active_particle_count"]),
        "active_particle_fraction": float(totals["active_particle_count"] / total_gas_count),
        "total_gas_mass_Msun": total_gas_mass,
        "active_mass_Msun": active_mass,
        "active_mass_fraction": float(active_mass / total_gas_mass),
        "oxygen_renormalisation_scale": oxygen_scale,
        "cosmic_ray_ionization_rate_s-1": float(
            settings["chemistry"]["cosmic_ray_ionization_rate_s-1"]
        ),
        "surrogate_model": str(model_path),
        "surrogate_sha256": model_sha,
        "particle_level_validation_passed": particle_validation_passed,
        "projected_scope_validation_passed": True,
        "projected_closure_report": str(closure_path),
        "projected_closure_sha256": closure_sha,
        "quantitative_scope": settings["chemistry"]["quantitative_scope"],
        "high_resolution_transfer_gate_passed": highres_transfer_gate_passed,
        "high_resolution_transfer_gate": {
            "maximum_uncovered_active_mass_fraction": maximum_uncovered_mass,
            "minimum_reliable_rare_line_luminosity_fraction": minimum_reliable_rare,
        },
        "training_coverage": {
            "policy": settings["chemistry"]["out_of_domain_policy"],
            "maximum_mahalanobis_distance_squared": maximum_coverage_distance2,
            "outside": {
                coverage_name: {
                    "particle_count": int(outside_count[coverage_index]),
                    "particle_fraction_of_active": float(
                        outside_count[coverage_index]
                        / max(totals["active_particle_count"], 1.0)
                    ),
                    "mass_Msun": float(outside_mass[coverage_index]),
                    "mass_fraction_of_active": float(
                        coverage_mass_fraction[coverage_index]
                    ),
                    "line_luminosity_fraction": {
                        name: float(coverage_luminosity_fraction[coverage_index, line_index])
                        for line_index, name in enumerate(("CO", "13CO", "C18O"))
                    },
                }
                for coverage_index, coverage_name in enumerate(COVERAGE_NAMES)
            },
        },
        "mass_weighted_abundance_active": {
            name: float(abundance_mass_sum[index] / max(active_mass, 1.0e-300))
            for index, name in enumerate(("CO", "13CO", "C18O"))
        },
        "total_line_luminosity_erg_s": {
            name: float(luminosity_sum[index])
            for index, name in enumerate(("CO", "13CO", "C18O"))
        },
        "integrated_intensity_ratios": {
            "13CO_over_CO": float(np.sum(intensity["13CO"]) / np.sum(intensity["CO"])),
            "C18O_over_CO": float(np.sum(intensity["C18O"]) / np.sum(intensity["CO"])),
            "13CO_over_C18O": float(
                np.sum(intensity["13CO"]) / np.sum(intensity["C18O"])
            ),
        },
        "positive_pixel_W_K_km_s": {
            name: positive_summary(values) for name, values in intensity.items()
        },
        "maps_npz": str(maps_path),
        "elapsed_merge_seconds": float(time.perf_counter() - started),
    }
    summary_path = resolve_path(output_paths["summary_json"], config_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    plot_maps(
        resolve_path(output_paths["figure_png"], config_path),
        abundance,
        intensity,
        int(settings["snapshot"]["index"]),
    )
    plot_co(
        resolve_path(output_paths["co_figure_png"], config_path),
        h2_co,
        intensity["CO"],
        int(settings["snapshot"]["index"]),
    )
    print(f"Saved merged maps: {maps_path}", flush=True)
    print(f"Saved summary: {summary_path}", flush=True)
    print(
        f"High-resolution transfer gate passed: {highres_transfer_gate_passed}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("context", "part", "merge"), required=True)
    parser.add_argument("--part-index", type=int)
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    settings = load_settings(config_path)
    if args.stage == "context":
        prepare_context(config_path, settings)
    elif args.stage == "part":
        if args.part_index is None:
            raise ValueError("--part-index is required for stage=part")
        process_part(config_path, settings, args.part_index)
    else:
        merge_parts(config_path, settings)


if __name__ == "__main__":
    main()
