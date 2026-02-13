import os
import pandas as pd
import matplotlib.pyplot as plt

from .config import PLOT_DIR, SAVE_PLOTS, FIGSIZE, DPI


plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "axes.titlesize": 13,
    "axes.titlepad": 12,
    "font.size": 10,
})


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def finish_map(ax, title=None, aoi_4326=None, districts_4326=None, pad=0.01):
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)

    if districts_4326 is not None:
        districts_4326.plot(ax=ax, facecolor="none", edgecolor="lightgray", linewidth=0.8)

    if aoi_4326 is not None:
        aoi_4326.boundary.plot(ax=ax, linewidth=2)
        minx, miny, maxx, maxy = aoi_4326.total_bounds
        dx = (maxx - minx) * pad
        dy = (maxy - miny) * pad
        ax.set_xlim(minx - dx, maxx + dx)
        ax.set_ylim(miny - dy, maxy + dy)


def show_or_save(fig, name=None):
    if SAVE_PLOTS and name is not None:
        ensure_dir(PLOT_DIR)
        out = os.path.join(PLOT_DIR, f"{name}.png")
        fig.savefig(out, bbox_inches="tight")
    plt.show()


def annotate_polygons(gdf_poly_4326, label_col, ax, fontsize=8):
    tmp = gdf_poly_4326[[label_col, "geometry"]].copy()
    tmp["pt"] = tmp.geometry.representative_point()
    for _, r in tmp.iterrows():
        if pd.isna(r[label_col]):
            continue
        ax.text(
            r["pt"].x, r["pt"].y, str(r[label_col]),
            ha="center", va="center", fontsize=fontsize
        )


def new_fig_ax(figsize=FIGSIZE):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


