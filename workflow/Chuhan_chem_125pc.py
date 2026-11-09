"""
Chuhan_chem_125pc.py

Read cached chemistry maps, convolve each 2D map to 125 pc Gaussian
resolution, and calculate luminosity-weighted radial line ratios.
"""

import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter


OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR_OVERRIDE",
    os.path.join(os.path.dirname(__file__), "outputs"),
)
SNAPSHOT_INDICES = [
    int(item)
    for item in os.environ.get("SNAPSHOT_INDICES", "600").replace(",", " ").split()
]

MAP_SIZE_PC = 30000.0
TARGET_RESOLUTION_PC = 125.0
GAUSSIAN_TRUNCATE = 4.0
PC_CM = 3.0856775814913673e18
K_B = 1.380649e-16
C_LIGHT = 2.99792458e10
LINE_FREQUENCIES_HZ = {
    "CO": 115271201800.0,
    "13CO": 110201354300.0,
    "C18O": 109782173400.0,
}

RADIAL_BINS_KPC = np.arange(0.0, 16.0, 1.0)
FIXED_RADII_KPC = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])


def finite_gaussian_filter(image, sigma_pix, **kwargs):
    """Gaussian smooth with NaN-safe normalization."""
    image = np.asarray(image, dtype=float)
    if sigma_pix <= 0:
        return image.copy()

    finite = np.isfinite(image)
    values = np.where(finite, image, 0.0)
    weights = finite.astype(float)

    smooth_values = gaussian_filter(values, sigma=sigma_pix, **kwargs)
    smooth_weights = gaussian_filter(weights, sigma=sigma_pix, **kwargs)

    smoothed = np.full_like(image, np.nan, dtype=float)
    valid = smooth_weights > 0
    smoothed[valid] = smooth_values[valid] / smooth_weights[valid]
    return smoothed


def smooth_to_125pc(image, pixel_size_pc):
    """Convolve a map with a Gaussian whose FWHM is 125 pc."""
    sigma_pc = TARGET_RESOLUTION_PC / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_pix = sigma_pc / pixel_size_pc
    return finite_gaussian_filter(
        image,
        sigma_pix=sigma_pix,
        mode="constant",
        cval=0.0,
        truncate=GAUSSIAN_TRUNCATE,
    )


def ratio_map(numerator, denominator):
    ratio = np.full_like(numerator, np.nan, dtype=float)
    mask = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0)
    ratio[mask] = numerator[mask] / denominator[mask]
    return ratio


def luminosity_to_w(map_luminosity, frequency_hz):
    """Convert erg s^-1 pc^-2 to Rayleigh-Jeans K km s^-1."""
    intensity = np.asarray(map_luminosity, dtype=float) / (4.0 * np.pi * PC_CM**2)
    return intensity * C_LIGHT**3 / (2.0 * K_B * frequency_hz**3 * 1.0e5)


def radial_ratio_of_sums(
    numerator,
    denominator,
    radius_bins_kpc,
    extent_kpc,
    sector_count=12,
):
    """Return sum(numerator)/sum(denominator) and azimuthal jackknife errors."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    nx, ny = numerator.shape
    dx = (extent_kpc[1] - extent_kpc[0]) / nx
    dy = (extent_kpc[3] - extent_kpc[2]) / ny
    x = extent_kpc[0] + (np.arange(nx) + 0.5) * dx
    y = extent_kpc[2] + (np.arange(ny) + 0.5) * dy
    xx, yy = np.meshgrid(x, y, indexing="ij")
    radius = np.hypot(xx, yy)
    angle = np.mod(np.arctan2(yy, xx), 2.0 * np.pi)
    sector = np.floor(angle / (2.0 * np.pi / sector_count)).astype(int)
    finite = (
        np.isfinite(numerator) & np.isfinite(denominator)
        & (numerator >= 0.0) & (denominator >= 0.0)
    )

    size = len(radius_bins_kpc) - 1
    profile = np.full(size, np.nan)
    error = np.full(size, np.nan)
    pixel_count = np.zeros(size, dtype=np.int64)
    for index in range(size):
        annulus = (
            finite
            & (radius >= radius_bins_kpc[index])
            & (radius < radius_bins_kpc[index + 1])
        )
        pixel_count[index] = np.count_nonzero(annulus)
        denominator_sum = np.sum(denominator[annulus])
        if denominator_sum <= 0.0:
            continue
        profile[index] = np.sum(numerator[annulus]) / denominator_sum
        leave_one_out = []
        for sector_index in range(sector_count):
            selected = annulus & (sector != sector_index)
            local_denominator = np.sum(denominator[selected])
            if local_denominator > 0.0:
                leave_one_out.append(
                    np.sum(numerator[selected]) / local_denominator
                )
        leave_one_out = np.asarray(leave_one_out, dtype=float)
        if leave_one_out.size >= 2:
            mean = np.mean(leave_one_out)
            error[index] = np.sqrt(
                (leave_one_out.size - 1.0) / leave_one_out.size
                * np.sum((leave_one_out - mean) ** 2)
            )

    radii_mid = 0.5 * (radius_bins_kpc[:-1] + radius_bins_kpc[1:])
    return radii_mid, profile, error, pixel_count


def radial_profile_from_map(map_data, radius_bins_kpc, extent_kpc):
    """Compute azimuthal mean/std radial profiles from a square 2D map."""
    nx, ny = map_data.shape
    x = np.linspace(extent_kpc[0], extent_kpc[1], nx)
    y = np.linspace(extent_kpc[2], extent_kpc[3], ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    rr = np.sqrt(xx**2 + yy**2)

    profile_mean = np.full(len(radius_bins_kpc) - 1, np.nan, dtype=float)
    profile_std = np.full(len(radius_bins_kpc) - 1, np.nan, dtype=float)
    for idx in range(len(profile_mean)):
        mask = (rr >= radius_bins_kpc[idx]) & (rr < radius_bins_kpc[idx + 1])
        values = map_data[mask]
        values = values[np.isfinite(values)]
        if values.size > 0:
            profile_mean[idx] = np.nanmean(values)
            profile_std[idx] = np.nanstd(values)

    radii_mid = 0.5 * (radius_bins_kpc[:-1] + radius_bins_kpc[1:])
    return radii_mid, profile_mean, profile_std


def sample_profile_at_radii(radii_mid, profile, target_radii_kpc):
    """Interpolate a radial profile at target radii."""
    valid = np.isfinite(profile) & np.isfinite(radii_mid)
    if np.count_nonzero(valid) < 2:
        return np.full(len(target_radii_kpc), np.nan, dtype=float)
    return np.interp(
        target_radii_kpc,
        radii_mid[valid],
        profile[valid],
        left=np.nan,
        right=np.nan,
    )


def load_cached_maps(snapshot_index):
    maps_infile = os.path.join(
        OUTPUT_DIR, f"snapshot_{snapshot_index:03d}_maps_chuhan.npz"
    )
    if not os.path.exists(maps_infile):
        raise FileNotFoundError(
            f"Cached map file not found: {maps_infile}. "
            "Run Chuhan_chem.py first to create the npz cache."
        )

    with np.load(maps_infile) as data:
        maps = {key: data[key] for key in data.files}
    return maps_infile, maps


def smooth_cached_maps(maps):
    res = int(np.asarray(maps.get("res", maps["mapLCO"].shape[0])).item())
    pixel_size_pc = MAP_SIZE_PC / res
    sigma_pc = TARGET_RESOLUTION_PC / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_pix = sigma_pc / pixel_size_pc

    smooth_keys = [
        "NHmap",
        "CO_abundance",
        "C13O_abundance",
        "C18O_abundance",
        "H2_COMap",
        "mapLCO",
        "mapL13CO",
        "mapLC18O",
        "mapYSO",
    ]

    smoothed = dict(maps)
    for key in smooth_keys:
        if key in maps:
            smoothed[key] = smooth_to_125pc(maps[key], pixel_size_pc)

    smoothed["WCO"] = luminosity_to_w(
        smoothed["mapLCO"], LINE_FREQUENCIES_HZ["CO"]
    )
    smoothed["W13CO"] = luminosity_to_w(
        smoothed["mapL13CO"], LINE_FREQUENCIES_HZ["13CO"]
    )
    smoothed["WC18O"] = luminosity_to_w(
        smoothed["mapLC18O"], LINE_FREQUENCIES_HZ["C18O"]
    )
    smoothed["mapR13"] = ratio_map(smoothed["W13CO"], smoothed["WCO"])
    smoothed["mapR18"] = ratio_map(smoothed["WC18O"], smoothed["WCO"])
    smoothed["mapR13C18"] = ratio_map(smoothed["W13CO"], smoothed["WC18O"])
    smoothed["res"] = res
    smoothed["pixel_size_pc"] = pixel_size_pc
    smoothed["target_resolution_pc"] = TARGET_RESOLUTION_PC
    smoothed["gaussian_sigma_pc"] = sigma_pc
    smoothed["gaussian_sigma_pix"] = sigma_pix
    return smoothed


def save_smoothed_maps(snapshot_index, maps):
    out_file = os.path.join(
        OUTPUT_DIR, f"snapshot_{snapshot_index:03d}_maps_chuhan_125pc.npz"
    )
    save_keys = [
        "NHmap",
        "CO_abundance",
        "C13O_abundance",
        "C18O_abundance",
        "H2_COMap",
        "mapLCO",
        "mapL13CO",
        "mapLC18O",
        "mapR13",
        "mapR18",
        "mapR13C18",
        "WCO",
        "W13CO",
        "WC18O",
        "mapYSO",
        "res",
        "pixel_size_pc",
        "target_resolution_pc",
        "gaussian_sigma_pc",
        "gaussian_sigma_pix",
        "stellar_yield_mode",
        "include_old_stellar_yields",
        "oxygen_renorm_enabled",
        "oxygen_renorm_mode",
        "oxygen_renorm_target_o16_c12",
        "oxygen_renorm_scale_median",
        "direct_method",
        "direct_input_path",
        "direct_merged_path",
        "direct_active_particle_count",
        "direct_total_snapshot_particle_count",
        "direct_converged_count",
        "direct_unconverged_count",
        "lookup_active_particle_count",
        "lookup_particle_validation_passed",
        "lookup_projected_scope_validation_passed",
        "lookup_highres_transfer_gate_passed",
        "lookup_quantitative_scope",
        "lookup_minimum_validated_radius_kpc",
        "lookup_maximum_validated_radius_kpc",
        "lookup_projected_closure_report",
        "lookup_projected_closure_sha256",
        "cosmic_ray_ionization_rate_s_1",
        "surrogate_sha256",
        "workflow_config_path",
        "workflow_config_sha256",
        "workflow_config_json",
    ]
    np.savez(out_file, **{key: maps[key] for key in save_keys if key in maps})
    return out_file


def plot_ratio(snapshot_index, maps):
    extent = [-15, 15, -15, 15]
    radii, profile, error, _ = radial_ratio_of_sums(
        maps["W13CO"], maps["WC18O"], RADIAL_BINS_KPC, extent
    )
    fig_ratio, ax_ratio = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
    ax_ratio.errorbar(
        radii,
        profile,
        yerr=error,
        marker="o",
        markersize=4,
        linewidth=1.4,
        capsize=2,
    )
    minimum_validated_radius = float(
        np.asarray(maps.get("lookup_minimum_validated_radius_kpc", 0.0)).item()
    )
    maximum_validated_radius = float(
        np.asarray(maps.get("lookup_maximum_validated_radius_kpc", 15.0)).item()
    )
    if minimum_validated_radius > 0.0:
        ax_ratio.axvspan(0.0, minimum_validated_radius, color="0.85", zorder=0)
    if maximum_validated_radius < 15.0:
        ax_ratio.axvspan(maximum_validated_radius, 15.0, color="0.85", zorder=0)
    ax_ratio.set_xlim(0.0, 15.0)
    ax_ratio.set_xlabel("Galactocentric radius [kpc]")
    ax_ratio.set_ylabel(r"$\sum W_{13}/\sum W_{18}$")
    ax_ratio.set_title(
        f"Snapshot {snapshot_index:03d}: stacking-equivalent ratio, 125 pc"
    )
    ax_ratio.grid(alpha=0.25)
    ratio_fig_path = os.path.join(
        OUTPUT_DIR, f"13CO_over_C18O_ratio_snap{snapshot_index:03d}_125pc.png"
    )
    fig_ratio.savefig(ratio_fig_path, dpi=200)
    plt.close(fig_ratio)
    return ratio_fig_path


def plot_maps(snapshot_index, maps):
    extent = [-15, 15, -15, 15]
    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.22, wspace=0.18)
    panels = [
        ("CO abundance-weighted surface", np.log10(maps["CO_abundance"] + 1e-30), "viridis", r"$\log_{10}(\Sigma_g X_{\rm CO})$"),
        ("13CO abundance-weighted surface", np.log10(maps["C13O_abundance"] + 1e-30), "viridis", r"$\log_{10}(\Sigma_g X_{13})$"),
        ("C18O abundance-weighted surface", np.log10(maps["C18O_abundance"] + 1e-30), "viridis", r"$\log_{10}(\Sigma_g X_{18})$"),
        ("CO integrated intensity", np.log10(maps["WCO"] + 1e-30), "inferno", r"$\log_{10} W$ [K km s$^{-1}$]"),
        ("13CO integrated intensity", np.log10(maps["W13CO"] + 1e-30), "inferno", r"$\log_{10} W$ [K km s$^{-1}$]"),
        ("C18O integrated intensity", np.log10(maps["WC18O"] + 1e-30), "inferno", r"$\log_{10} W$ [K km s$^{-1}$]"),
    ]
    for idx, (title, image, cmap, colorbar_label) in enumerate(panels):
        row = 0 if idx < 3 else 1
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(image, origin="lower", extent=extent, cmap=cmap, aspect="equal")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label(colorbar_label)
    fig.suptitle(
        f"Snapshot {snapshot_index:03d} - CO / 13CO / C18O maps, 125 pc",
        y=0.995,
    )
    fig_path = os.path.join(
        OUTPUT_DIR, f"co_isotope_maps_snap{snapshot_index:03d}_125pc.png"
    )
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_radial_profile(snapshot_index, maps):
    extent = [-15, 15, -15, 15]
    ratio_profile_radii, ratio_profile, ratio_profile_error, pixel_count = (
        radial_ratio_of_sums(
            maps["W13CO"], maps["WC18O"], RADIAL_BINS_KPC, extent
        )
    )
    fixed_radius_values = sample_profile_at_radii(
        ratio_profile_radii, ratio_profile, FIXED_RADII_KPC
    )
    fixed_radius_err = sample_profile_at_radii(
        ratio_profile_radii, ratio_profile_error, FIXED_RADII_KPC
    )

    out_file = os.path.join(
        OUTPUT_DIR,
        f"snapshot_{snapshot_index:03d}_radial_profile_13CO_over_C18O_125pc.npz",
    )
    np.savez(
        out_file,
        snapshot_index=snapshot_index,
        radii_kpc=ratio_profile_radii,
        ratio_13CO_over_C18O=ratio_profile,
        ratio_13CO_over_C18O_std=ratio_profile_error,
        ratio_13CO_over_C18O_jackknife_error=ratio_profile_error,
        annulus_pixel_count=pixel_count,
        radial_edges_kpc=RADIAL_BINS_KPC,
        profile_statistic="sum(W13CO) / sum(WC18O) in 1 kpc annuli",
        jackknife_sector_count=12,
        lookup_highres_transfer_gate_passed=maps.get(
            "lookup_highres_transfer_gate_passed", False
        ),
        lookup_quantitative_scope=maps.get("lookup_quantitative_scope", ""),
        lookup_minimum_validated_radius_kpc=maps.get(
            "lookup_minimum_validated_radius_kpc", 0.0
        ),
        lookup_maximum_validated_radius_kpc=maps.get(
            "lookup_maximum_validated_radius_kpc", 15.0
        ),
        fixed_radii_kpc=FIXED_RADII_KPC,
        ratio_at_fixed_radii=fixed_radius_values,
        ratio_at_fixed_radii_err=fixed_radius_err,
        target_resolution_pc=TARGET_RESOLUTION_PC,
        gaussian_sigma_pc=maps["gaussian_sigma_pc"],
        gaussian_sigma_pix=maps["gaussian_sigma_pix"],
        pixel_size_pc=maps["pixel_size_pc"],
        stellar_yield_mode=maps.get("stellar_yield_mode", "all"),
        include_old_stellar_yields=maps.get(
            "include_old_stellar_yields", True
        ),
    )
    return out_file


def main():
    print("=" * 70)
    print("Chuhan_chem_125pc.py - cached map smoothing test")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for snapshot_index in SNAPSHOT_INDICES:
        print(f"\n处理快照 {snapshot_index:03d}")
        maps_infile, maps = load_cached_maps(snapshot_index)
        print(f"读取缓存: {maps_infile}")

        smoothed_maps = smooth_cached_maps(maps)
        print(
            "高斯平滑: "
            f"FWHM={TARGET_RESOLUTION_PC:.1f} pc, "
            f"sigma={smoothed_maps['gaussian_sigma_pc']:.2f} pc "
            f"= {smoothed_maps['gaussian_sigma_pix']:.2f} pixel"
        )

        smooth_npz = save_smoothed_maps(snapshot_index, smoothed_maps)
        radial_npz = save_radial_profile(snapshot_index, smoothed_maps)
        ratio_png = plot_ratio(snapshot_index, smoothed_maps)
        maps_png = plot_maps(snapshot_index, smoothed_maps)

        print(f"保存 125pc map npz: {smooth_npz}")
        print(f"保存 125pc 径向曲线: {radial_npz}")
        print(f"保存 125pc 比值图: {ratio_png}")
        print(f"保存 125pc 六面板图: {maps_png}")


if __name__ == "__main__":
    main()
