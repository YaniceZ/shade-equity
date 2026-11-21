"""Publication figures (183 mm double-column, Arial 7 pt, vector PDF + PNG)."""

from __future__ import annotations

import json

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D

from . import config as C
from . import data as D
from . import surfaces
from .analysis import edge_table
from .network import suns
from .shadow import sunlit_raster

MM = 1 / 25.4
FULL = 183 * MM
HALF = 89 * MM

INK = "#1a1a19"
INK_2 = "#52514e"
RULE = "#d9d7d0"
LAND = "#efeee9"
WATER = "#dfe9ef"
BLD = "#8c8a85"
LIDAR = "#1f5fa8"  # reference
GLOBAL = "#d1612b"  # open global data
TREE = "#2f7d32"
# Shade (0 = full sun, 1 = full shade): light warm sun to deep blue shade.
SHADE_CMAP = LinearSegmentedColormap.from_list("shade", ["#f6c445", "#f0e3b2", "#9cc2d9", "#2f6ea5", "#12325c"])
SUN_CMAP = LinearSegmentedColormap.from_list("sun", ["#fff6dc", "#f6c445", "#e0712c", "#a8321b"])
HEIGHT_CMAP = LinearSegmentedColormap.from_list("h", ["#f7f7f2", "#bdbab2", "#6e6b66", "#2b2a28"])
TREE_CMAP = LinearSegmentedColormap.from_list("t", ["#d9ead3", "#7fb069", "#2f7d32", "#173f19"])
KIND_LABEL = {"grocery": "Grocery", "pharmacy": "Pharmacy", "clinic": "Clinic or doctor", "transit": "Transit stop",
              "park": "Park", "community": "Community centre"}


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 600,
    })


def panel(ax, letter, x=-0.14, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def save(fig, name: str) -> None:
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    out = C.FIGURES / name
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def mapax(ax):
    ax.set_axis_off()
    ax.set_aspect("equal")


def scalebar(ax, x, y, length=1000, label="1 km", h=None):
    h = h or length * 0.04
    ax.add_patch(plt.Rectangle((x, y), length, h, color=INK, lw=0))
    ax.text(x + length / 2, y + h * 1.8, label, ha="center", va="bottom", fontsize=6)


def north(ax, x, y, size=300):
    ax.annotate("N", xy=(x, y + size), xytext=(x, y), ha="center", va="bottom", fontsize=6.5, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color=INK, shrinkA=0, shrinkB=0))


def _results(name):
    with open(C.RESULTS / name, encoding="utf-8") as fh:
        return json.load(fh)


def _window(city, cx, cy, half):
    g = D.grid(city)
    x0, _, _, y1 = g["bounds"]
    c0, c1 = int(cx - half - x0), int(cx + half - x0)
    r0, r1 = int(y1 - (cy + half)), int(y1 - (cy - half))
    return (r0, r1, c0, c1), (cx - half, cx + half, cy - half, cy + half)


# Zoom area: Kitsilano / Fairview boundary — mixed towers, low-rise and mature street trees.
ZOOM = {"vancouver": (489_400, 5_456_900, 300)}


def land(city):
    """City land outline for display (local areas exclude the water inside the municipal boundary)."""
    nb = D.neighbourhoods(city)
    return gpd.GeoDataFrame(geometry=[nb.geometry.union_all()], crs=city.crs) if nb is not None else D.boundary(city)


def title(ax, text, letter=None):
    if letter:
        ax.text(0.0, 1.015, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")
    ax.text(0.07 if letter else 0.0, 1.015, text, transform=ax.transAxes, fontsize=7, va="bottom", ha="left")


def map_extent(ax, gdf, pad=150):
    xmin, ymin, xmax, ymax = gdf.total_bounds
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    return xmin, ymin, xmax, ymax


def hbar(fig, ax, cmap, vmin, vmax, label, ticks=None, fmt=None, rect=(0.02, -0.07, 0.42, 0.025)):
    cax = ax.inset_axes(list(rect))
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax), cmap=cmap), cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.3)
    cb.ax.tick_params(length=2, width=0.4, labelsize=6)
    if ticks is not None:
        cb.set_ticks(ticks, labels=fmt or [f"{t:g}" for t in ticks])
    cax.set_title(label, fontsize=6.5, pad=2, loc="left")
    return cb


def fig1(city):
    """Study area and the two obstacle models."""
    t = edge_table(city)
    nb = D.neighbourhoods(city)
    ld = land(city)
    S = surfaces.load(city)
    cx, cy, half = ZOOM[city.key]
    win, ext = _window(city, cx, cy, half)
    r0, r1, c0, c1 = win
    (h, az, el), = [s for s in suns(city) if s[0] == C.PEAK_HOUR]

    fig = plt.figure(figsize=(FULL, 108 * MM))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1], wspace=0.08, hspace=0.42)
    ax = fig.add_subplot(gs[:, :2])
    mapax(ax)
    ld.plot(ax=ax, color=LAND, lw=0)
    t = t[t.geometry.centroid.within(ld.geometry.iloc[0].buffer(30))].sort_values("shade_lidar")
    t.plot(ax=ax, column="shade_lidar", cmap=SHADE_CMAP, vmin=0, vmax=1, lw=0.3)
    nb.boundary.plot(ax=ax, color="white", lw=0.6)
    ax.plot([ext[0], ext[1], ext[1], ext[0], ext[0]], [ext[2], ext[2], ext[3], ext[3], ext[2]], color=INK, lw=0.9)
    xmin, ymin, xmax, ymax = map_extent(ax, ld)
    scalebar(ax, xmax - 2300, ymin + 150, 2000, "2 km")
    title(ax, "Walkway shade, 10:00–17:00 (LiDAR)", "a")
    hbar(fig, ax, SHADE_CMAP, 0, 1, "Share of time in shade", ticks=[0, 0.25, 0.5, 0.75, 1], fmt=["0", "", "0.5", "", "1"], rect=(0.03, 0.06, 0.36, 0.022))

    def obstacles(axi, bk, vk, text, letter):
        mapax(axi)
        b = S[bk][r0:r1, c0:c1]
        v = S[vk][r0:r1, c0:c1]
        axi.imshow(np.where(v > 0, v, np.nan), extent=ext, cmap=TREE_CMAP, vmin=0, vmax=25, interpolation="nearest")
        axi.imshow(np.where(b > 0, b, np.nan), extent=ext, cmap=HEIGHT_CMAP, vmin=0, vmax=50, interpolation="nearest")
        title(axi, text, letter)

    def shade(axi, bk, vk, text, letter):
        mapax(axi)
        s = sunlit_raster(S["ground"], S[bk], S[vk], az, el, C.CELL, C.TREE_TRANSMISSIVITY, C.TRUNK_RATIO, window=win)
        axi.imshow(1 - s, extent=ext, cmap=SHADE_CMAP, vmin=0, vmax=1, interpolation="nearest")
        b = S[bk][r0:r1, c0:c1]
        axi.imshow(np.where(b > 0, 1, np.nan), extent=ext, cmap=ListedColormap([BLD]), interpolation="nearest")
        title(axi, text, letter)

    ab = fig.add_subplot(gs[0, 2])
    obstacles(ab, "ref_bld", "ref_veg", f"LiDAR {city.lidar_year}", "b")
    ac = fig.add_subplot(gs[0, 3])
    obstacles(ac, "glb_bld", "glb_veg", "GlobalBuildingAtlas + Meta/WRI", "c")
    ad = fig.add_subplot(gs[1, 2])
    shade(ad, "ref_bld", "ref_veg", f"Shade at {C.PEAK_HOUR:.0f}:00, LiDAR", "d")
    ae = fig.add_subplot(gs[1, 3])
    shade(ae, "glb_bld", "glb_veg", f"Shade at {C.PEAK_HOUR:.0f}:00, global data", "e")
    w = ext[1] - ext[0]
    ad.add_patch(plt.Rectangle((ext[0] + 0.03 * w, ext[2] + 0.03 * w), 0.42 * w, 0.13 * w, color="white", alpha=0.85, lw=0))
    scalebar(ad, ext[0] + 0.06 * w, ext[2] + 0.06 * w, 200, "200 m", h=0.012 * w)
    hbar(fig, ab, HEIGHT_CMAP, 0, 50, "Building height (m)", ticks=[0, 25, 50], rect=(0.0, -0.13, 0.42, 0.03))
    hbar(fig, ac, TREE_CMAP, 0, 25, "Tree height (m)", ticks=[0, 10, 20], rect=(0.0, -0.13, 0.42, 0.03))
    handles = [Line2D([], [], marker="s", ls="", color=BLD, ms=4, label="Building"),
               Line2D([], [], marker="s", ls="", color="#12325c", ms=4, label="Shade"),
               Line2D([], [], marker="s", ls="", color="#f6c445", ms=4, label="Direct sun")]
    ae.legend(handles=handles, loc="upper left", bbox_to_anchor=(-1.02, -0.02), ncol=3, handletextpad=0.3, columnspacing=1.0)
    save(fig, "fig1_study_area")


def _hexbin(ax, x, y, lim, label_x, label_y, gridsize=60, w=None):
    hb = ax.hexbin(x, y, C=w, reduce_C_function=np.sum if w is not None else None, gridsize=gridsize, extent=(lim[0], lim[1], lim[0], lim[1]),
                   bins="log", cmap=LinearSegmentedColormap.from_list("b", ["#eef2f7", "#7aa4cf", "#1f5fa8", "#0b2a52"]), mincnt=1, lw=0)
    ax.plot(lim, lim, color=INK_2, lw=0.5, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_aspect("equal")
    return hb


def fig2(city):
    """How well open global data reproduce LiDAR shade."""
    v = _results("validation.json")
    bh = surfaces.building_heights(city)
    bh = bh[(bh["lidar_median"] >= 2.5) & (bh["n"] >= 20) & bh["gba_height"].notna()]
    t = edge_table(city)
    fig = plt.figure(figsize=(FULL, 112 * MM))
    gs = fig.add_gridspec(2, 3, wspace=0.55, hspace=0.6)

    ax = fig.add_subplot(gs[0, 0])
    _hexbin(ax, bh["lidar_median"], bh["gba_height"], (0, 80), "LiDAR building height (m)", "GlobalBuildingAtlas height (m)")
    b = v["buildings"]["all"]
    ax.text(0.04, 0.96, f"n = {b['n']:,}\nbias {b['bias']:+.1f} m\nMAE {b['mae']:.1f} m", transform=ax.transAxes, va="top", fontsize=6)
    panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    cv = v["canopy"]
    bars = [cv["lidar_cover"], cv["green"]["lidar_cover"], cv["meta_cover"]]
    ax.bar([0, 1, 2], bars, color=[LIDAR, "#7aa4cf", GLOBAL], width=0.62)
    for i, val in enumerate(bars):
        ax.text(i, val + 0.006, f"{val:.0%}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks([0, 1, 2], ["LiDAR\nall objects", "LiDAR\ngreen only", "Meta/WRI"])
    ax.set_ylabel("Canopy ≥2.5–3 m outside buildings")
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    ax.set_ylim(0, 0.55)
    ax.text(0.98, 0.98, f"Meta finds {cv['green']['recall']:.0%} of\ngreen LiDAR canopy\n({cv['green']['recall_ge8m']:.0%} of crowns ≥8 m)",
            transform=ax.transAxes, ha="right", va="top", fontsize=6)
    panel(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    _hexbin(ax, t["shade_lidar"], t["shade_global"], (0, 1), "Walkway shade, LiDAR", "Walkway shade, global data", gridsize=40, w=t["length"] / 1000)
    e = v["network"]["edge_daytime"]
    ax.text(0.04, 0.96, f"{v['network']['km']:,.0f} km of walkway\nbias {e['bias']:+.2f}, MAE {e['mae']:.2f}\nSpearman ρ = {e['spearman']:.2f}",
            transform=ax.transAxes, va="top", fontsize=6, bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="none", alpha=0.9))
    panel(ax, "c")

    ax = fig.add_subplot(gs[1, 0])
    di = v["network"]["diurnal"]
    style_map = {"lidar": (LIDAR, "-", "LiDAR"), "global": (GLOBAL, "-", "Global data"),
                 "global_bld+lidar_veg": ("#8f6bb0", "--", "Global buildings + LiDAR trees"),
                 "lidar_bld+global_veg": (TREE, "--", "LiDAR buildings + global trees")}
    for s, (col, ls, lab) in style_map.items():
        hrs = sorted(float(k) for k in di[s])
        ax.plot(hrs, [di[s][f"{h:g}"] for h in hrs], color=col, ls=ls, lw=1, label=lab)
    band = {}
    for ref, tag in (("lidar_green", "_green"), ("lidar_clean", "_clean")):
        f = D.city_dir(city) / f"edge_sun{tag}.parquet"
        if f.exists():
            es = pd.read_parquet(f)
            es = es[es.eid.isin(t.index)].assign(length=lambda x: t["length"].reindex(x.eid).to_numpy())
            band[ref] = es.groupby("hour").apply(lambda g: 1 - np.average(g["sun"], weights=g["length"]), include_groups=False)
    if "lidar_green" in band:
        hrs = sorted(float(k) for k in di["lidar"])
        ax.fill_between(hrs, band["lidar_green"].reindex(hrs), [di["lidar"][f"{h:g}"] for h in hrs], color=LIDAR, alpha=0.12, lw=0,
                        label="LiDAR reference range")
    if "lidar_clean" in band:
        ax.plot(band["lidar_clean"].index, band["lidar_clean"].to_numpy(), color=LIDAR, ls=":", lw=1, label="LiDAR, cleaned")
    ax.set_xlabel("Local time, 28 June 2021")
    ax.set_ylabel("Shaded share of walkways")
    ax.set_xticks([10, 12, 14, 16], ["10:00", "12:00", "14:00", "16:00"])
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1, fontsize=6)
    panel(ax, "d")

    ax = fig.add_subplot(gs[1, 1])
    dc = v["network"]["decomposition"]
    names = ["building_error", "tree_error", "interaction", "total_error"]
    labels = ["Buildings", "Trees", "Interaction", "Total"]
    vals = [100 * dc[n] for n in names]
    cols = ["#8f6bb0", TREE, "#b5b2aa", GLOBAL]
    ax.barh(range(4)[::-1], vals, color=cols, height=0.6)
    ax.axvline(0, color=INK_2, lw=0.5)
    for i, val in zip(range(4)[::-1], vals, strict=True):
        ax.text(val + (0.3 if val >= 0 else -0.3), i, f"{val:+.1f}", va="center", ha="left" if val >= 0 else "right", fontsize=6)
    ax.set_yticks(range(4)[::-1], labels)
    ax.set_xlabel("Error in city shade share\n(percentage points, global − LiDAR)")
    lo = min(vals + [0]) * 1.5 - 1
    hi = max(vals + [0]) * 1.5 + 1
    ax.set_xlim(lo, hi)
    panel(ax, "e", -0.35)

    ax = fig.add_subplot(gs[1, 2])
    nbs = pd.read_csv(C.RESULTS / "neighbourhood_shade.csv", index_col=0)
    ax.scatter(nbs["shade_lidar"], nbs["shade_global"], s=9, color=GLOBAL, edgecolor="white", lw=0.3, zorder=3)
    lo, hi = nbs[["shade_lidar", "shade_global"]].min().min() - 0.03, nbs[["shade_lidar", "shade_global"]].max().max() + 0.03
    ax.plot([lo, hi], [lo, hi], color=INK_2, lw=0.5, ls="--")
    for i, name in enumerate(nbs["shade_lidar"].nsmallest(2).index.tolist() + nbs["shade_lidar"].nlargest(2).index.tolist()):
        ax.annotate(name, (nbs.loc[name, "shade_lidar"], nbs.loc[name, "shade_global"]), fontsize=5.5,
                    xytext=[(4, -6), (4, 3), (-4, 4), (4, -5)][i], ha="right" if i == 2 else "left", textcoords="offset points")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Neighbourhood shade, LiDAR")
    ax.set_ylabel("Neighbourhood shade, global data")
    a = v["areas"]["neighbourhoods"]
    ax.text(0.04, 0.96, f"22 local areas\nρ = {a['spearman']:.2f}\nbias {a['bias']:+.3f}", transform=ax.transAxes, va="top", fontsize=6)
    panel(ax, "f")
    save(fig, "fig2_validation")


def fig3(city):
    """Older adults' sun exposure on the way to essential destinations."""
    eq = _results("equity.json")
    cell = pd.read_parquet(D.city_dir(city) / "cell_summary.parquet")
    pop = D.population(city)
    ld = land(city)
    nb = D.neighbourhoods(city)
    fig = plt.figure(figsize=(FULL, 128 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1.1, 1], wspace=0.42, hspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    mapax(ax)
    ld.plot(ax=ax, color=LAND, lw=0)
    g = gpd.GeoDataFrame(cell.join(pop[["geometry"]]), geometry="geometry", crs=city.crs)
    g = g[(g["pop65"] >= 1) & g.geometry.centroid.within(ld.geometry.iloc[0])]
    vmax = 14.0
    g.plot(ax=ax, column="sunmin", cmap=SUN_CMAP, vmin=0, vmax=vmax, lw=0)
    nb.boundary.plot(ax=ax, color="white", lw=0.5)
    xmin, ymin, xmax, ymax = map_extent(ax, ld)
    scalebar(ax, xmax - 2300, ymin + 150, 2000, "2 km")
    title(ax, f"Minutes in direct sun per trip, residents 65+, {C.PEAK_HOUR:.0f}:00", "a")
    hbar(fig, ax, SUN_CMAP, 0, vmax, "Mean of six destination types (LiDAR)", ticks=[0, 5, 10, 14], fmt=["0", "5", "10", "≥14"],
         rect=(0.03, 0.06, 0.36, 0.022))

    kinds = [k for k in KIND_LABEL if k in eq["lidar"]]
    y = np.arange(len(kinds))[::-1]
    ax = fig.add_subplot(gs[0, 1])
    for off, sc, col, lab in ((0.15, "lidar", LIDAR, "LiDAR"), (-0.15, "global", GLOBAL, "Global data")):
        vals = np.array([eq[sc][k]["sun_min_older_peak_65"] for k in kinds])
        ci = np.array([eq[sc][k]["sun_min_older_peak_65_ci"] for k in kinds])
        ax.errorbar(vals, y + off, xerr=[vals - ci[:, 0], ci[:, 1] - vals], fmt="o", ms=3, color=col, lw=0.8, capsize=0, label=lab)
    ax.set_yticks(y, [KIND_LABEL[k] for k in kinds])
    ax.set_xlabel(f"Minutes in direct sun per one-way trip\n(65+, {C.SPEED_OLDER} m/s, {C.PEAK_HOUR:.0f}:00; 95% block-bootstrap CI)")
    ax.set_xlim(0, 13.5)
    ax.legend(loc="upper left")
    panel(ax, "b", -0.42, 1.0)

    ax = fig.add_subplot(gs[1, 0])
    near = [eq["lidar"][k]["near_10min_65"] for k in kinds]
    shaded = [eq["lidar"][k]["near_and_shaded_65"] for k in kinds]
    shaded_g = [eq["global"][k]["near_and_shaded_65"] for k in kinds]
    ax.barh(y, near, color="#dcdad3", height=0.7, label="Nearest one within 10 min walk")
    ax.barh(y + 0.17, shaded, color=LIDAR, height=0.34, label="…and route ≥50% shaded (LiDAR)")
    ax.barh(y - 0.17, shaded_g, color=GLOBAL, height=0.34, label="…and route ≥50% shaded (global data)")
    ax.set_yticks(y, [KIND_LABEL[k] for k in kinds])
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlabel(f"Share of residents aged 65+ ({C.PEAK_HOUR:.0f}:00)")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.3), ncol=1, fontsize=6)
    panel(ax, "c", -0.32, 1.0)

    ax = fig.add_subplot(gs[1, 1])
    lv = eq["lidar"]
    steps = [("sun_min_all_adult_speed", "All residents at 1.3 m/s", "#dcdad3"), ("sun_min_all_older_speed", "All residents at 0.9 m/s", "#7aa4cf"),
             ("sun_min_older_peak_65", "Residents 65+ at 0.9 m/s", LIDAR)]
    hgt = 0.26
    for j, (key, lab, col) in enumerate(steps):
        ax.barh(y + (1 - j) * hgt, [lv[k][key] for k in kinds], height=hgt, color=col, label=lab)
    ax.set_yticks(y, [KIND_LABEL[k] for k in kinds])
    ax.set_xlabel(f"Minutes in direct sun per trip ({C.PEAK_HOUR:.0f}:00, LiDAR)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.3), ncol=1, fontsize=6)
    panel(ax, "d", -0.42, 1.0)
    save(fig, "fig3_older_adults")


def fig4(city):
    """Two ways to cut older adults' sun exposure: route choice and targeted street trees."""
    eq = _results("equity.json")
    sc = pd.read_csv(C.RESULTS / "planting_scenarios.csv")
    flow = pd.read_parquet(D.city_dir(city) / "older_flow.parquet")["older_flow"]
    t = edge_table(city)
    fig = plt.figure(figsize=(FULL, 70 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.45)

    ax = fig.add_subplot(gs[0, 0])
    ss = eq["shade_seeking"]
    al = sorted(ss, key=float)
    ax.plot([0] + [ss[a]["detour_pct_65"] for a in al], [0] + [ss[a]["sun_reduction_pct_65"] for a in al], "-o", color=LIDAR, ms=3, lw=1)
    for a in al:
        ax.annotate(f"α={a}", (ss[a]["detour_pct_65"], ss[a]["sun_reduction_pct_65"]), fontsize=5.5, xytext=(3, -6), textcoords="offset points")
    ax.set_xlabel("Extra walking distance (%)")
    ax.set_ylabel("Less time in direct sun (%)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_title("Shade-seeking routes, 65+", fontsize=7)
    panel(ax, "a", -0.3)

    ax = fig.add_subplot(gs[0, 1])
    base = sc.loc[sc.rule == "baseline", "sun_min_older"].iloc[0]
    for rule, col, lab in (("targeted", LIDAR, "Where older adults walk"), ("sunniest", "#9a9893", "Sunniest streets")):
        s = sc[sc.rule == rule].sort_values("budget_km")
        ax.plot([0] + s["budget_km"].tolist(), [0] + (100 * (1 - s["sun_min_older"] / base)).tolist(), "-o", ms=3, lw=1, color=col, label=lab)
    ax.set_xlabel("Streets planted (km)")
    ax.set_ylabel("Less time in direct sun, 65+ (%)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.legend(loc="upper left", fontsize=6)
    ax.set_title("Street-tree scenarios (simulated)", fontsize=7)
    panel(ax, "b", -0.3)

    ax = fig.add_subplot(gs[0, 2])
    mapax(ax)
    ld = land(city)
    ld.plot(ax=ax, color=LAND, lw=0)
    t = t.join(flow.rename("flow"))
    t["load"] = t["flow"].fillna(0)
    t = t[t.geometry.centroid.within(ld.geometry.iloc[0].buffer(30))]
    t[t["load"] > 0].sort_values("load").plot(ax=ax, column="load", cmap=LinearSegmentedColormap.from_list("f", ["#e4e2dc", "#8d8b86", "#3a3936"]),
                                              norm=mpl.colors.LogNorm(vmin=10, vmax=t["load"].quantile(0.995)), lw=0.35)
    pl = D.city_dir(city) / "planted_targeted.parquet"
    if pl.exists():
        sel = pd.read_parquet(pl)["eid"]
        t[t.index.isin(sel)].plot(ax=ax, color=LIDAR, lw=1.4)
    xmin, ymin, xmax, ymax = map_extent(ax, ld)
    scalebar(ax, xmax - 2300, ymin + 150, 2000, "2 km")
    title(ax, "Older adults routed per street (grey) and\nthe 50 km planted in the targeted scenario (blue)", "c")
    save(fig, "fig4_interventions")


def all_figures(city):
    style()
    for f in (fig1, fig2, fig3, fig4):
        try:
            f(city)
            D.log(f"  {f.__name__} saved")
        except FileNotFoundError as exc:
            D.log(f"  {f.__name__} skipped: {exc}")

