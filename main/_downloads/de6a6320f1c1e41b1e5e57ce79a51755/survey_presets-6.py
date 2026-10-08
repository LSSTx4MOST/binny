import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np

from binny import NZTomography


def plot_bins(ax, result, title):
    z = result.z
    bin_dict = result.bins
    keys = sorted(bin_dict.keys())

    colors = cmr.take_cmap_colors(
        "viridis",
        len(keys),
        cmap_range=(0.1, 0.9),
        return_fmt="hex",
    )

    for i, (color, key) in enumerate(zip(colors, keys, strict=True)):
        curve = np.asarray(bin_dict[key], dtype=float)

        ax.fill_between(
            z,
            0.0,
            curve,
            color=color,
            alpha=0.65,
            linewidth=0.0,
            zorder=10 + i,
        )

        ax.plot(
            z,
            curve,
            color="k",
            linewidth=1.8,
            zorder=20 + i,
        )

    ax.plot(z, np.zeros_like(z), color="k", linewidth=2.0, zorder=1000)

    ax.set_title(title)
    ax.set_xlabel("Redshift $z$")


tomo = NZTomography()
result = tomo.build_survey_bins(
    "4most_crs_bg",
    role="source",
    sample="bg",
    include_tomo_metadata=True,
)

edges = result.tomo_meta["bins"]["bin_edges"]


fig, axes = plt.subplots(
    1,
    2,
    figsize=(11.5, 4.5),
)

ax = axes[0]
ax.fill_between(
    result.z,
    0.0,
    result.nz,
    color=plt.get_cmap("viridis")(0.5),
    alpha=0.65,
    linewidth=0.0,
    zorder=10,
)
ax.plot(result.z, result.nz, color="k", linewidth=1.8, zorder=20)

for edge in edges:
    ax.axvline(edge, color="k", linestyle="--", linewidth=1.0, alpha=0.6, zorder=5)

ax.plot(result.z, np.zeros_like(result.z), color="k", linewidth=2.0, zorder=1000)
ax.set_title("4MOST-CRS BG: parent $n(z)$ (Smail fit)")
ax.set_xlabel("Redshift $z$")
ax.set_ylabel(r"Normalized $n(z)$")

plot_bins(axes[1], result, "4MOST-CRS BG: five spec-z bins")
axes[1].set_ylabel(r"Normalized $n_i(z)$")

for ax in axes:
    ax.set_xlim(0.0, 0.5)

plt.suptitle("4MOST-CRS BG survey preset tomography", fontsize=16)

plt.tight_layout(rect=(0, 0, 1, 0.95))