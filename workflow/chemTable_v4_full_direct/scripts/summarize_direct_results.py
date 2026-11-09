#!/usr/bin/env python3
"""Summarize snapshot-600 full-direct chemistry, opacity, and mock detections."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from workflow_config import config_path, load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG, CONFIG_PATH, CONFIG_SHA256, _ = load_config()
INPUT_PATH = config_path(CONFIG, "direct_input")
RESULT_PATH = config_path(CONFIG, "merged_result")
OUTPUT_DIR = config_path(CONFIG, "map_output_dir")
OUTPUT_PATH = OUTPUT_DIR / "direct_science_summary.json"
OPACITY_FIGURE_PATH = OUTPUT_DIR / "direct_optical_depth_luminosity_cdf.png"

REFERENCE = CONFIG["chemistry"]["reference_abundances_per_H_nucleus"]
DIAGNOSTICS = CONFIG["diagnostics"]
DEFAULTS = {
    "x12C": REFERENCE["12C"],
    "x13C": REFERENCE["13C"],
    "x16O": REFERENCE["16O"],
    "x18O": REFERENCE["18O"],
    "X_CO": REFERENCE["CO"],
    "X_13CO": REFERENCE["13CO"],
    "X_C18O": REFERENCE["C18O"],
}
LINE_NAMES = ("CO", "13CO", "C18O")


def weighted_mean(values, weights, mask):
    good = mask & np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(good):
        return None
    return float(np.average(values[good], weights=weights[good]))


def weighted_quantiles(values, weights, quantiles, mask):
    good = (
        mask & np.isfinite(values) & np.isfinite(weights)
        & (weights > 0.0) & (values >= 0.0)
    )
    if not np.any(good):
        return [None] * len(quantiles)
    values = values[good]
    weights = weights[good]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= np.sum(weights)
    return [float(np.interp(q, cumulative, values)) for q in quantiles]


def phase_summary(fields, inputs, mass, mask):
    result = {
        "particle_count": int(np.count_nonzero(mask)),
        "mass_fraction_of_active": float(np.sum(mass[mask]) / np.sum(mass)),
    }
    for name in ("x12C", "x13C", "x16O", "x18O"):
        result[name] = weighted_mean(inputs[name], mass, mask)
    for name in ("X_H2", "X_CO", "X_13CO", "X_C18O"):
        result[name] = weighted_mean(fields[name], mass, mask)
    result["elemental_13C_over_18O"] = result["x13C"] / result["x18O"]
    result["molecular_13CO_over_C18O"] = result["X_13CO"] / result["X_C18O"]
    for name in DEFAULTS:
        value = result[name]
        result[f"{name}_over_default"] = value / DEFAULTS[name]
        result[f"{name}_offset_dex"] = float(np.log10(value / DEFAULTS[name]))
    return result


def opacity_summary(tau, luminosity, converged):
    positive_luminosity = converged & np.isfinite(luminosity) & (luminosity > 0.0)
    q50, q90, q99 = weighted_quantiles(
        tau, luminosity, (0.5, 0.9, 0.99), positive_luminosity
    )
    total = float(np.sum(luminosity[positive_luminosity]))
    result = {
        "luminosity_weighted_tau_p50": q50,
        "luminosity_weighted_tau_p90": q90,
        "luminosity_weighted_tau_p99": q99,
    }
    for threshold in DIAGNOSTICS["opacity_thresholds"]:
        selected = positive_luminosity & (tau > threshold)
        tag = str(threshold).replace(".", "p")
        result[f"luminosity_fraction_tau_gt_{tag}"] = (
            float(np.sum(luminosity[selected]) / total) if total > 0.0 else None
        )
    return result


def plot_opacity_cdf(fields, converged):
    colours = {"CO": "#2f4b7c", "13CO": "#d95f02", "C18O": "#1b9e77"}
    labels = {"CO": r"$^{12}$CO", "13CO": r"$^{13}$CO", "C18O": r"C$^{18}$O"}
    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    for name in LINE_NAMES:
        tau = fields[f"tau_{name}_1_0"]
        luminosity = fields[f"L_{name}_1_0"]
        good = (
            converged & np.isfinite(tau) & (tau >= 0.0)
            & np.isfinite(luminosity) & (luminosity > 0.0)
        )
        values = np.maximum(tau[good], 1.0e-8)
        weights = luminosity[good]
        order = np.argsort(values)
        values = values[order]
        cumulative = np.cumsum(weights[order]) / np.sum(weights)
        keep = np.unique(np.linspace(0, values.size - 1, 5000, dtype=int))
        ax.plot(
            values[keep], cumulative[keep], color=colours[name],
            linewidth=2.0, label=labels[name],
        )
    ax.axvline(1.0, color="0.35", linestyle="--", linewidth=1.2, label=r"$\tau=1$")
    ax.set_xscale("log")
    ax.set_xlim(1.0e-5, 1.0e3)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Particle LVG optical depth, $\tau_{1-0}$")
    ax.set_ylabel("Cumulative fraction of line luminosity")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(OPACITY_FIGURE_PATH, dpi=220)
    plt.close(fig)


def mock_summary(path):
    with np.load(path) as data:
        w13 = np.asarray(data["W13_hex_K_km_s"], dtype=float)
        w18 = np.asarray(data["W18_hex_K_km_s"], dtype=float)
        det = np.asarray(data["det"], dtype=bool)
        lower = np.asarray(data["lower"], dtype=bool)
        upper = np.asarray(data["upper"], dtype=bool)
        nondet = np.asarray(data["nondet"], dtype=bool)
        ratio = np.asarray(data["ratio_det"], dtype=float)
        finite_ratio = ratio[det & np.isfinite(ratio) & (ratio > 0.0)]
        return {
            "sigma13_K_km_s": float(data["sigma13_K_km_s"]),
            "sigma18_K_km_s": float(data["sigma18_K_km_s"]),
            "cell_count": int(w13.size),
            "detections": int(np.count_nonzero(det)),
            "lower_limits": int(np.count_nonzero(lower)),
            "upper_limits": int(np.count_nonzero(upper)),
            "non_detections": int(np.count_nonzero(nondet)),
            "detection_fraction": float(np.count_nonzero(det) / w13.size),
            "detected_ratio_median": (
                float(np.median(finite_ratio)) if finite_ratio.size else None
            ),
            "detected_ratio_p99": (
                float(np.percentile(finite_ratio, 99.0)) if finite_ratio.size else None
            ),
            "sum_intensity_ratio": float(np.sum(w13) / np.sum(w18)),
        }


def main():
    with h5py.File(INPUT_PATH, "r") as source:
        mass = np.asarray(source["mass_Msun"][:], dtype=float)
        inputs = {
            name: np.asarray(source[name][:], dtype=float)
            for name in ("nH_cm3", "x12C", "x13C", "x16O", "x18O")
        }
        oxygen_scale = float(source.attrs["oxygen_renorm_scale"])
    with h5py.File(RESULT_PATH, "r") as source:
        status = np.asarray(source["status"][:], dtype=np.int8)
        fields = {
            name: np.asarray(source[name][:], dtype=float)
            for name in (
                "Tg_K", "X_H2", "X_CO", "X_13CO", "X_C18O",
                "L_CO_1_0", "L_13CO_1_0", "L_C18O_1_0",
                "tau_CO_1_0", "tau_13CO_1_0", "tau_C18O_1_0",
            )
        }

    converged = status > 0
    h2_limit = float(DIAGNOSTICS["h2_dominated_threshold_2XH2"])
    co_limit = float(DIAGNOSTICS["co_dominated_threshold_XCO_over_x12C"])
    dense_limit = float(DIAGNOSTICS["dense_gas_threshold_nH_cm-3"])
    h2_dominated = converged & (2.0 * fields["X_H2"] > h2_limit)
    co_dominated = converged & (fields["X_CO"] / inputs["x12C"] > co_limit)
    dense = converged & (inputs["nH_cm3"] >= dense_limit)
    phases = {
        "all_active_converged": converged,
        "h2_dominated_2XH2_gt_0p5": h2_dominated,
        "co_dominated_XCO_over_x12C_gt_0p5": co_dominated,
        "dense_nH_ge_100_cm3": dense,
    }
    summary = {
        "workflow_config_path": str(CONFIG_PATH),
        "workflow_config_sha256": CONFIG_SHA256,
        "active_particle_count": int(status.size),
        "converged_particle_count": int(np.count_nonzero(converged)),
        "unconverged_particle_count": int(np.count_nonzero(~converged)),
        "convergence_fraction": float(np.count_nonzero(converged) / status.size),
        "oxygen_renormalisation_scale": oxygen_scale,
        "phase_definitions": {
            "all_active_converged": "valid neutral in-footprint particles solved by DESPOTIC",
            "h2_dominated_2XH2_gt_0p5": f"2 X(H2) > {h2_limit:g}",
            "co_dominated_XCO_over_x12C_gt_0p5": f"X(CO)/x(12C) > {co_limit:g}",
            "dense_nH_ge_100_cm3": f"nH >= {dense_limit:g} cm^-3",
        },
        "phases": {
            name: phase_summary(fields, inputs, mass, mask)
            for name, mask in phases.items()
        },
        "line_optical_depth": {
            name: opacity_summary(
                fields[f"tau_{name}_1_0"], fields[f"L_{name}_1_0"], converged
            )
            for name in LINE_NAMES
        },
        "mock_observations": {},
    }
    for path in sorted(OUTPUT_DIR.glob("R13_18_mock_limits_*.npz")):
        summary["mock_observations"][path.name] = mock_summary(path)

    plot_opacity_cdf(fields, converged)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
