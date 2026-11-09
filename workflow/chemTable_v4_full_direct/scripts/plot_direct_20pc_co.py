#!/usr/bin/env python3
"""Redraw snapshot 600 20 pc CO surface-density and intensity figures."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

from workflow_config import config_path, load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG, CONFIG_PATH, CONFIG_SHA256, _ = load_config()
PROJECTION = CONFIG["projection"]
DEFAULT_OUTDIR = config_path(CONFIG, "map_output_dir")

MAP_SIZE_PC = float(PROJECTION["map_size_pc"])
PC_CM = 3.0856775814913673e18
MSUN_G = 1.98847e33
MH_G = 1.6735575e-24
HYDROGEN_MASS_FRACTION = float(CONFIG["snapshot"]["hydrogen_mass_fraction"])
K_BOLTZMANN = 1.380649e-16
C_LIGHT = 2.99792458e10
NU_CO_10_HZ = float(PROJECTION["line_frequencies_Hz"]["CO"])


def luminosity_surface_density_to_k_kms(luminosity_surface_density):
    line_surface_brightness = luminosity_surface_density / (4.0 * np.pi * PC_CM**2)
    return line_surface_brightness * C_LIGHT**3 / (2.0 * K_BOLTZMANN * NU_CO_10_HZ**3 * 1.0e5)


def surface_density_to_column(sigma_msun_pc2, molecular_weight):
    sigma_g_cm2 = sigma_msun_pc2 * MSUN_G / PC_CM**2
    return sigma_g_cm2 / (molecular_weight * MH_G)


def summarize(name, values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    pos = finite[finite > 0.0]
    lines = []
    if finite.size:
        lines.append(
            f"  all min/median/p90/p99/max = "
            f"{np.nanmin(finite):.8g} {np.nanmedian(finite):.8g} "
            f"{np.nanpercentile(finite, 90):.8g} {np.nanpercentile(finite, 99):.8g} "
            f"{np.nanmax(finite):.8g}"
        )
    else:
        lines.append("  all min/median/p90/p99/max = nan nan nan nan nan")
    if pos.size:
        lines.append(
            f"  positive p16/median/p84/p99/max = "
            f"{np.nanpercentile(pos, 16):.8g} {np.nanmedian(pos):.8g} "
            f"{np.nanpercentile(pos, 84):.8g} {np.nanpercentile(pos, 99):.8g} "
            f"{np.nanmax(pos):.8g}"
        )
    else:
        lines.append("  positive p16/median/p84/p99/max = nan nan nan nan nan")
    lines.append(f"  positive pixels = {pos.size}")
    return f"{name}:\n" + "\n".join(lines)


def positive_log_norm(values, dynamic_range=1.0e4):
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        return LogNorm(vmin=1.0e-4, vmax=1.0)
    vmax = float(np.nanpercentile(positive, 99.0))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.nanmax(positive))
    vmin = max(float(np.nanmin(positive)), vmax / dynamic_range)
    return LogNorm(vmin=vmin, vmax=vmax)


def percentile_log_norm(values, percentiles, vmin=None, vmax=None):
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        return LogNorm(vmin=1.0e-4, vmax=1.0)
    percentile_vmin, percentile_vmax = np.nanpercentile(positive, percentiles)
    if vmin is None:
        vmin = percentile_vmin
    if vmax is None:
        vmax = percentile_vmax
    return LogNorm(vmin=float(vmin), vmax=float(vmax))


def display_cmap(name):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad("white")
    cmap.set_under("0.92")
    return cmap


def log_colorbar_extend(is_log, full_log_range, log_percentiles):
    if not is_log or full_log_range:
        return "neither"
    if log_percentiles is None:
        return "both"
    lower, upper = log_percentiles
    if lower > 0.0 and upper < 100.0:
        return "both"
    if lower > 0.0:
        return "min"
    if upper < 100.0:
        return "max"
    return "neither"


def save_two_panel(
    path,
    left,
    right,
    left_title,
    right_title,
    left_label,
    right_label,
    left_vmin=0.0,
    left_vmax=100.0,
    right_vmin=0.0,
    right_vmax=None,
    left_log=False,
    right_log=False,
    full_log_range=False,
    log_percentiles=None,
):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    extent = [-15, 15, -15, 15]
    left_plot = (
        np.where(np.isfinite(left) & (left > 0.0), left, np.nan)
        if left_log else left
    )
    if left_log and left_vmin is not None and left_vmax is not None:
        left_norm = LogNorm(vmin=left_vmin, vmax=left_vmax)
    elif left_log:
        if log_percentiles is not None:
            left_norm = percentile_log_norm(
                left_plot, log_percentiles, vmin=left_vmin, vmax=left_vmax
            )
        else:
            left_norm = LogNorm() if full_log_range else positive_log_norm(left_plot)
    else:
        left_norm = Normalize(vmin=left_vmin, vmax=left_vmax)
    im0 = axes[0].imshow(
        left_plot,
        origin="lower",
        extent=extent,
        cmap=display_cmap("magma"),
        norm=left_norm,
    )
    right_plot = np.where(np.isfinite(right) & (right > 0.0), right, np.nan) if right_log else right
    if right_log and right_vmin is not None and right_vmax is not None:
        right_norm = LogNorm(vmin=right_vmin, vmax=right_vmax)
    elif right_log:
        if log_percentiles is not None:
            right_norm = percentile_log_norm(
                right_plot, log_percentiles, vmin=right_vmin, vmax=right_vmax
            )
        else:
            right_norm = LogNorm() if full_log_range else positive_log_norm(right_plot)
    else:
        right_norm = Normalize(vmin=right_vmin, vmax=right_vmax)
    im1 = axes[1].imshow(
        right_plot,
        origin="lower",
        extent=extent,
        cmap=display_cmap("inferno"),
        norm=right_norm,
    )
    axes[0].set_title(left_title)
    axes[1].set_title(right_title)
    for ax in axes:
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")
        ax.set_aspect("equal")
    fig.colorbar(
        im0, ax=axes[0], label=left_label, fraction=0.046, pad=0.04,
        extend=log_colorbar_extend(left_log, full_log_range, log_percentiles),
    )
    fig.colorbar(
        im1, ax=axes[1], label=right_label, fraction=0.046, pad=0.04,
        extend=log_colorbar_extend(right_log, full_log_range, log_percentiles),
    )
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_one_panel(path, image, title, label, vmax=100.0):
    fig, ax = plt.subplots(figsize=(5.8, 5.0), constrained_layout=True)
    extent = [-15, 15, -15, 15]
    im = ax.imshow(image, origin="lower", extent=extent, cmap="magma", vmin=0.0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_full_direct_30kpc(path, sigma_h2_co, wco):
    """Plot the strict full-direct CO products with matched logarithmic ranges."""
    alpha_co = float(PROJECTION["assumed_alpha_CO_Msun_pc-2_per_K_km_s-1"])
    sigma_min = 1.0e-3
    sigma_max = 1.0e2
    extent = [-MAP_SIZE_PC / 2000.0, MAP_SIZE_PC / 2000.0] * 2

    h2_plot = np.where(
        np.isfinite(sigma_h2_co) & (sigma_h2_co > 0.0), sigma_h2_co, np.nan
    )
    wco_plot = np.where(np.isfinite(wco) & (wco > 0.0), wco, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), constrained_layout=True)
    im0 = axes[0].imshow(
        h2_plot,
        origin="lower",
        extent=extent,
        cmap="magma",
        norm=LogNorm(vmin=sigma_min, vmax=sigma_max),
    )
    im1 = axes[1].imshow(
        wco_plot,
        origin="lower",
        extent=extent,
        cmap="magma",
        norm=LogNorm(vmin=sigma_min / alpha_co, vmax=sigma_max / alpha_co),
    )
    axes[0].set_title(r"CO-inferred H$_2$ surface density")
    axes[1].set_title(r"$^{12}$CO(1-0) integrated intensity")
    for axis in axes:
        axis.set_xlabel("x [kpc]")
        axis.set_ylabel("y [kpc]")
        axis.set_aspect("equal")
    fig.colorbar(
        im0,
        ax=axes[0],
        label=r"$\Sigma_{\mathrm{H_2,CO}}$ [$M_\odot\,\mathrm{pc}^{-2}$]",
        fraction=0.046,
        pad=0.04,
    )
    fig.colorbar(
        im1,
        ax=axes[1],
        label=r"$W_{\mathrm{CO}(1-0)}$ [K km s$^{-1}$]",
        fraction=0.046,
        pad=0.04,
    )
    fig.suptitle(
        r"Snapshot 600: per-particle DESPOTIC, "
        r"$\mathtt{smallabd}=10^{-15}$, auto-$t_{\rm EqGuess}$"
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--maps", type=Path, default=None)
    parser.add_argument(
        "--strict-only",
        action="store_true",
        help="Only write the strict smallabd=1e-15, auto-tEqGuess 30 kpc figure.",
    )
    args = parser.parse_args()
    outdir = args.outdir
    maps_path = args.maps or outdir / "snapshot_600_maps_chuhan.npz"
    outdir.mkdir(parents=True, exist_ok=True)

    with np.load(maps_path) as data:
        map_lco = data["mapLCO"].astype(float)
        h2_co = data["H2_COMap"].astype(float)
        co_abundance_surface = data["CO_abundance"].astype(float)
        res = int(np.asarray(data.get("res", map_lco.shape[0])).item())

    pixel_size_pc = MAP_SIZE_PC / res
    wco = luminosity_surface_density_to_k_kms(map_lco)
    sigma_h2_co = h2_co
    sigma_12co = co_abundance_surface * HYDROGEN_MASS_FRACTION * 28.0
    n_12co = surface_density_to_column(sigma_12co, 28.0)
    alpha_co = np.full_like(
        wco, float(PROJECTION["assumed_alpha_CO_Msun_pc-2_per_K_km_s-1"])
    )

    set_chem_eq = CONFIG["chemistry"]["setChemEq"]
    if not np.isclose(float(set_chem_eq["small_abundance"]), 1.0e-15):
        raise RuntimeError("The requested strict plot requires small_abundance=1e-15")
    if set_chem_eq["tEqGuess_s"] is not None:
        raise RuntimeError("The requested strict plot requires automatic tEqGuess")

    strict_path = (
        outdir
        / "snapshot_600_full_direct_smallabd_1e-15_auto_30kpc_COinferredH2_WCO10.png"
    )
    save_full_direct_30kpc(
        strict_path,
        sigma_h2_co,
        wco,
    )
    if args.strict_only:
        print("Saved", strict_path)
        return

    save_two_panel(
        outdir / "snapshot_600_12CO_surface_density_and_intensity_20pc.png",
        sigma_12co,
        wco,
        r"$^{12}$C$^{16}$O surface density",
        r"$^{12}$CO(1-0) integrated intensity",
        r"$\Sigma_{^{12}C^{16}O}$ [$M_\odot\,pc^{-2}$]",
        r"$W_{12CO}$ [K km s$^{-1}$]",
        left_vmax=100.0,
        right_log=True,
    )
    stats_12co = [
        f"Input: {maps_path}",
        f"Pixel size: {pixel_size_pc:.3f} pc",
        "Surface-density colorbar: 0 to 100 Msun pc^-2",
        summarize("Sigma_12C16O [Msun pc^-2]", sigma_12co),
        summarize("N_12C16O [cm^-2]", n_12co),
        summarize("W_12CO10 [K km s^-1]", wco),
        summarize("alpha_CO [Msun pc^-2 (K km s^-1)^-1]", alpha_co),
    ]
    (outdir / "snapshot_600_12CO_surface_density_and_intensity_20pc_stats.txt").write_text("\n".join(stats_12co) + "\n")

    save_two_panel(
        outdir / "snapshot_600_CO_surface_density_and_intensity_20pc.png",
        sigma_h2_co,
        wco,
        r"CO-inferred H$_2$ surface density",
        r"$^{12}$CO(1-0) integrated intensity",
        r"$\Sigma_{H2,CO}$ [$M_\odot\,pc^{-2}$]",
        r"$W_{12CO}$ [K km s$^{-1}$]",
        left_vmin=None,
        left_vmax=1.0e2,
        right_vmin=None,
        right_vmax=1.0e1,
        left_log=True,
        right_log=True,
        log_percentiles=(50.0, 99.9),
    )
    stats_co = [
        f"Input: {maps_path}",
        f"Pixel size: {pixel_size_pc:.3f} pc",
        "Logarithmic colorbars start at the positive-pixel median and end at "
        "1e2 Msun pc^-2 and 1e1 K km s^-1, respectively.",
        summarize("Sigma_H2_CO [Msun pc^-2]", sigma_h2_co),
        summarize("W_12CO10 [K km s^-1]", wco),
    ]
    (outdir / "snapshot_600_CO_surface_density_and_intensity_20pc_stats.txt").write_text("\n".join(stats_co) + "\n")

    save_one_panel(
        outdir / "snapshot_600_Hu2024_12CO_inferred_H2_surface_density_20pc.png",
        sigma_h2_co,
        r"Hu2024-style CO-inferred H$_2$ surface density",
        r"$\Sigma_{H2,CO}$ [$M_\odot\,pc^{-2}$]",
        vmax=100.0,
    )
    h2_from_lco = map_lco * float(PROJECTION["CO_inferred_H2_Msun_per_erg_s"])
    diff = np.abs(sigma_h2_co - h2_from_lco)
    rel = diff / np.maximum(np.abs(sigma_h2_co), 1e-300)
    stats_hu = [
        "Hu2024 method: M_H2,CO/Msun = "
        f"{PROJECTION['CO_inferred_H2_Msun_per_erg_s']:.8g} "
        "L_CO/(erg/s), projected to face-on Sigma_H2,CO.",
        f"Input: {maps_path}",
        f"Pixel size: {pixel_size_pc:.3f} pc",
        "Colorbar: 0 to 100 Msun pc^-2",
        f"Max |H2_COMap - mapLCO*2.3e-29| = {np.nanmax(diff):.8g}",
        f"Max relative difference = {np.nanmax(rel[np.isfinite(rel)]):.8g}",
        summarize("Sigma_H2_CO [Msun pc^-2]", sigma_h2_co),
    ]
    (outdir / "snapshot_600_Hu2024_12CO_inferred_H2_surface_density_20pc_stats.txt").write_text("\n".join(stats_hu) + "\n")

    print("Redrew 20 pc CO figures in", outdir)


if __name__ == "__main__":
    main()
