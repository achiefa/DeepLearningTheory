from pathlib import Path

from matplotlib import rc
import matplotlib.pyplot as plt

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)

from yadlt.distribution import Distribution
from yadlt.utils import compute_distance

FONTSIZE = 20
LABELSIZE = 16
LEGENDSIZE = 16
TICKSIZE = 14
FIGSIZE = (10, 5)

PLOT_DIR = Path(__file__).parent.parent / "doc/plots"


def produce_pdf_plot(
    x_grid,
    grids: list[Distribution],
    normalize_to=1,
    filename="pdf_plot.pdf",
    title=None,
):
    ref_grid = grids[normalize_to - 1]
    fig, axs = plt.subplots(
        2,
        1,
        figsize=FIGSIZE,
        sharex=True,
        gridspec_kw={
            "hspace": 0.0,
            "height_ratios": [3, 1],
        },
    )

    # Absolute plot
    for grid in grids:
        pl = axs[0].plot(x_grid, grid.get_mean(), label=rf"$\textrm{{{grid.name}}}$")
        axs[0].fill_between(
            x_grid,
            grid.get_mean() - grid.get_std(),
            grid.get_mean() + grid.get_std(),
            alpha=0.5,
            color=pl[0].get_color(),
        )

        normalised_grid = grid / ref_grid.get_mean()
        axs[1].plot(x_grid, normalised_grid.get_mean(), color=pl[0].get_color())
        axs[1].fill_between(
            x_grid,
            normalised_grid.get_mean() - normalised_grid.get_std(),
            normalised_grid.get_mean() + normalised_grid.get_std(),
            alpha=0.5,
            color=pl[0].get_color(),
        )

    # Set the title
    if title is not None:
        fig.suptitle(title, fontsize=FONTSIZE)

    # Y labels
    axs[0].set_ylabel(r"$xT_3$", fontsize=FONTSIZE)
    axs[1].set_ylabel(
        r"$\textrm{Ratio to}$" + "\n" + r"$\textrm{trained}$", fontsize=FONTSIZE
    )
    # X labels
    axs[1].set_xlabel(r"$x$", fontsize=FONTSIZE)
    # Set the x scale
    axs[1].set_xscale("linear")
    # Set the x limits
    axs[1].set_xlim(1e-3, 1.0)
    # Set the y limits
    axs[1].set_ylim(1 - 0.1, 1 + 0.1)
    # Tick size
    axs[0].tick_params(labelsize=TICKSIZE)
    axs[1].tick_params(labelsize=TICKSIZE)
    # Legend
    axs[0].legend(fontsize=LEGENDSIZE)
    # Save the figure
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, bbox_inches="tight")


def produce_distance_plot(
    x_grid,
    grids: list[Distribution],
    normalize_to=1,
    filename="distance_plot.pdf",
    title=None,
):
    # Distance plot
    distances = compute_distance(grids, normalize_to=normalize_to)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for idx, distance_spec in enumerate(distances):
        name = distance_spec[0]
        distance = distance_spec[1]
        ax.plot(x_grid, distance, label=name)

    # Set the title
    if title is not None:
        fig.suptitle(title, fontsize=FONTSIZE)
    # Y labels
    ax.set_ylabel(r"$\textrm{Distance from trained}$", fontsize=FONTSIZE)
    # X labels
    ax.set_xlabel(r"$x$", fontsize=FONTSIZE)
    # Tick size
    ax.tick_params(labelsize=TICKSIZE)
    # Legend
    ax.legend(fontsize=LEGENDSIZE)
    # Save the figure
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, bbox_inches="tight")
