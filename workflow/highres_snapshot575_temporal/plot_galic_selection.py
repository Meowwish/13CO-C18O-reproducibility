"""Moment-map selection experiment; cached x8/575 maps only, no DESPOTIC."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Patch, Circle
import numpy as np
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output/galic_selection"
DEPTHS = (1, 10, 100, 1000, 10000, 100000, 1000000)
EDGES = np.arange(16.)


def lattice():
    spacing = 0.0625
    dy = spacing * np.sqrt(3) / 2
    xy = []
    for j, y in enumerate(np.arange(-15 + dy / 2, 15, dy)):
        x = np.arange(-15 + spacing / 2, 15, spacing) + (j % 2) * spacing / 2
        x = x[x < 15 - 0.01]
        xy.extend(zip(x, np.full(len(x), y)))
    return np.array(xy)


def sample(array, xy):
    assert array.shape == (1500, 1500), array.shape
    # Cached projections are indexed (x,y), with pixel centres at -14.99 kpc.
    return map_coordinates(array, ((xy + 15) / 0.02 - 0.5).T,
                           order=1, mode="nearest", prefilter=False)


def save(fig, name):
    fig.savefig(OUT / (name + ".png"), dpi=190)
    fig.savefig(OUT / (name + ".pdf"), dpi=160)
    plt.close(fig)


def maps(branch, xy, h, sfr, ratio, state, depth, w12):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.7), layout="constrained")
    colors = ("0.90", "0.63", "0.30")
    detected = state == 3
    for ax in axes:
        for code, color in enumerate(colors):
            use = state == code
            ax.scatter(*xy[use].T, s=1.7, marker="h", color=color,
                       linewidths=0, rasterized=True)
        if np.nanmin(w12) < 30 < np.nanmax(w12):
            grid = np.linspace(-14.99, 14.99, 1500)
            ax.contour(grid, grid, w12.T, levels=[30], colors="black", linewidths=.35)
        ax.add_patch(Circle((-13, -13), .0625, color="black"))
        ax.set(xlim=(-15, 15), ylim=(-15, 15), aspect="equal",
               xlabel="x [kpc]", ylabel="y [kpc]")
    im = axes[0].scatter(*xy[detected].T, c=ratio[detected], s=2.4,
                         marker="h", linewidths=0, cmap="viridis",
                         norm=Normalize(0, 40), rasterized=True)
    fig.colorbar(im, ax=axes[0], location="top", extend="max",
                 label=r"$I(^{13}\mathrm{CO}(1-0))/I(\mathrm{C}^{18}\mathrm{O}(1-0))$")
    valid_sfr = detected & np.isfinite(sfr) & (sfr > 0)
    zero = detected & ~valid_sfr
    axes[1].scatter(*xy[zero].T, s=2.4, marker="h", color="black", linewidths=0,
                    rasterized=True)
    im = axes[1].scatter(*xy[valid_sfr].T, c=sfr[valid_sfr], s=2.4,
                         marker="h", linewidths=0, cmap="magma",
                         norm=LogNorm(1e-4, 1), rasterized=True)
    fig.colorbar(im, ax=axes[1], location="top", extend="both",
                 label=r"$\Sigma_{\mathrm{SFR},10\,\mathrm{Myr}}$ [$M_\odot\,\mathrm{yr}^{-1}\,\mathrm{kpc}^{-2}$]")
    fig.suptitle(f"Snapshot 575 x8 | {branch} | 125 pc beam | depth D={depth:g}\n"
                 + rf"$\sigma_{{13}}={.163/depth:g},\ \sigma_{{18}}={.144/depth:g}$ K km s$^{{-1}}$; both lines S/N > 3",
                 fontsize=11)
    handles = [Patch(color=c, label=l) for c, l in zip(colors,
               ("Neither line detected", "Lower limit (13CO only)", "Upper limit (C18O only)"))]
    handles.append(Patch(color="black", label="Detected lines, zero SFR"))
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=9)
    save(fig, f"snapshot575_{branch}_ratio_SFR_depth{depth}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    xy = lattice()
    radius = np.hypot(*xy.T)
    with np.load(ROOT / "output/sfr/snapshot575_SFR_maps.npz") as f:
        sfr = sample(f["10Myr_125pc"], xy)
    branches = {}
    for branch, folder in (("Full", ROOT / "output"), ("SLUG", ROOT / "slug_only/output")):
        with np.load(folder / "snapshot_575_maps_125pc.npz") as f:
            branches[branch] = {k: sample(f[k], xy) for k in ("WCO", "W13CO", "WC18O")}
            branches[branch]["mapWCO"] = f["WCO"]
    # The same finite square field is retained for every branch and depth.
    common = np.ones(len(xy), dtype=bool)
    for h in branches.values():
        for key in ("WCO", "W13CO", "WC18O"):
            common &= np.isfinite(h[key]) & (h[key] >= 0)
    rows = []
    for branch, h in branches.items():
        w13, w18 = h["W13CO"], h["WC18O"]
        ratio = np.divide(w13, w18, out=np.full(len(xy), np.nan), where=w18 > 0)
        for depth in DEPTHS:
            s13, s18 = .163 / depth, .144 / depth
            d13, d18 = w13 > 3*s13, w18 > 3*s18
            dual = common & d13 & d18
            state = np.full(len(xy), -1, dtype=np.int8)
            state[common] = 0
            state[common & d13 & ~d18] = 1
            state[common & ~d13 & d18] = 2
            state[dual] = 3
            # Independent line errors; this is not a spectral-noise simulation.
            relvar = np.full(len(xy), np.inf)
            positive = common & (w13 > 0) & (w18 > 0)
            relvar[positive] = (s13/w13[positive])**2 + (s18/w18[positive])**2
            strict = dual & (relvar < 1/9)
            for selection, selected in (("dual", dual), ("dual_and_ratio_snr", strict)):
                for lo, hi in zip(EDGES[:-1], EDGES[1:]):
                    base = common & (radius >= max(lo, .5)) & (radius < hi)
                    use = base & selected
                    n = int(use.sum())
                    values = dict(median_ratio=np.nan, p16=np.nan, p84=np.nan,
                                  ratio_of_medians=np.nan, ratio_of_sums=np.nan)
                    if n >= 10:
                        values.update(median_ratio=float(np.median(ratio[use])),
                                      p16=float(np.percentile(ratio[use], 16)),
                                      p84=float(np.percentile(ratio[use], 84)),
                                      ratio_of_medians=float(np.median(w13[use])/np.median(w18[use])),
                                      ratio_of_sums=float(w13[use].sum()/w18[use].sum()))
                    rows.append(dict(branch=branch, selection=selection, depth=depth,
                                     sigma13=s13, sigma18=s18, r_lo=max(lo, .5), r_hi=hi,
                                     r_mid=(max(lo, .5)+hi)/2, selected=n,
                                     total=int(base.sum()), fraction=n/base.sum() if base.any() else np.nan,
                                     **values))
            maps(branch, xy, h, sfr, ratio, state, depth, h["mapWCO"])
            print(f"{branch} depth={depth}: dual={dual.sum()}, ratio-SNR cut={strict.sum()}", flush=True)
    with (OUT / "radial_statistics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    colors = plt.cm.plasma(np.linspace(.05, .90, len(DEPTHS)))
    for selection in ("dual", "dual_and_ratio_snr"):
        fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True,
                                  sharey="row", layout="constrained")
        for col, branch in enumerate(branches):
            for color, depth in zip(colors, DEPTHS):
                series = [r for r in rows if r["branch"] == branch and r["depth"] == depth
                          and r["selection"] == selection]
                for ax, key in zip(axes[:, col], ("median_ratio", "ratio_of_medians", "ratio_of_sums", "fraction")):
                    ax.plot([r["r_mid"] for r in series], [r[key] for r in series], "o-",
                            ms=3, color=color, label=f"D={depth:g}")
            axes[0, col].set_title(branch)
            for ax in axes[:, col]:
                ax.axvspan(0, 1, color="0.93", zorder=-10)
                ax.axvspan(8, 15, color="0.93", zorder=-10)
                ax.grid(alpha=.18)
                ax.set_xlim(0, 15)
            for ax in axes[:3, col]:
                ax.set_yscale("log")
            axes[-1, col].set(xlabel="Radius [kpc]", ylim=(0, 1))
        for ax, label in zip(axes[:, 0], ("Median of pixel ratios", "Ratio of selected medians",
                                         "Ratio of selected sums", "Selected fraction")):
            ax.set_ylabel(label)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=7, fontsize=9)
        cut = "Both lines S/N > 3" if selection == "dual" else "Both lines AND propagated ratio S/N > 3"
        fig.suptitle("Snapshot 575 x8 | 125 pc beam, 62.5 pc sampling | " + cut
                     + "\n" + r"$\sigma_{13}=0.163/D$, $\sigma_{18}=0.144/D$ K km s$^{-1}$", fontsize=11)
        save(fig, "snapshot575_noise_radial_" + selection)
    manifest = dict(snapshot=575, resolution="x8", beam_pc=125, sampling_pc=62.5,
                    sigma_reference_K_km_s=[.163, .144], depths=list(DEPTHS),
                    data="Cached beam-convolved noiseless moment maps; no random noise added",
                    sfr="10 Myr initial stellar target mass / time / area, at snapshot-575 positions",
                    radial_inner_cut_kpc=.5, validated_radial_range_kpc=[1, 8],
                    ratio_color_limits=[0, 40], sfr_color_limits=[1e-4, 1], minimum_bin_samples=10,
                    common_sample_count=int(common.sum()))
    (OUT / "settings.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (OUT / "README.md").write_text("""# Snapshot 575 x8: Galic-style selection diagnostics

`snapshot575_noise_radial_dual` uses both line S/N > 3 (Fig. 2 caption).
`snapshot575_noise_radial_dual_and_ratio_snr` additionally requires propagated
ratio S/N > 3, addressing the extra condition mentioned in the radial-figure
caption. Independent line errors are assumed for that propagation.
The original text is not fully consistent about this extra condition, so the
two interpretations are kept separate rather than silently combined.

All maps use the Fig. 2 dual-line criterion, three grey limit categories, and
the SAME mask for the ratio and SFR panels. A black 12CO contour is drawn at
30 K km/s wherever that level exists. The small black circle is the 125 pc beam.
Fixed colour ranges are used across branches and depths; colourbar extensions
show clipping. True zero SFR among detected lines is black, not a fabricated
positive floor. Maps include the centre, as in Fig. 2. Radial statistics exclude
R < 0.5 kpc to match the observational analysis cut, without implying an AGN in
the simulation. R < 1 and R >= 8 kpc remain exploratory (shaded in profiles).

D is an inverse rms factor relative to (0.163, 0.144) K km/s, an approximate
single-channel SWAN conversion, NOT a measured common moment-0 uncertainty.
D=10 recovers the previous (0.0163, 0.0144) starting thresholds; D=1e6 is that
starting pair divided by 1e5. Depth changes the selection, not intrinsic flux.
No random noise, 3D CO/HCN mask, or spectral stacking has been introduced.

Sampling is a fixed triangular lattice at half-beam separation, bilinearly
sampling already convolved (x,y)-indexed maps; it is not another beam convolution.
All branches and depths use the same finite nonnegative 30 kpc square footprint.
Adjacent samples are correlated; counts are not independent beam counts.
Fractions use that footprint within each radial bin, not an observed SWAN mask.

The top radial row is the median of individual detected-pixel ratios. Two
additional rows isolate the choice of statistic on exactly the same selected
pixels. Neither selected ratio of medians nor selected ratio of sums is the
paper's uncut field-of-view ratio of medians or a spectral-stack measurement.
Bins with fewer than 10 selected samples are omitted; this is a display
safeguard, not a claim of statistical precision. The CSV includes p16/p84
pixel scatter, not confidence intervals. No all-pixel reference curve is added.

SFR is a 10 Myr stellar birth-mass estimate, not the observed Halpha+24um tracer.
The face-on kpc coordinates and full simulation field are preserved; these are
Fig. 2-style maps, not a reprojection onto M51 or a recreation of its footprint.
""")


if __name__ == "__main__":
    main()
