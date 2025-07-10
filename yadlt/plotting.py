"""
This modules the plotting structures for the YADLT library.
It provides three different plotting functions:
1. `produce_pdf_plot`: Produces a plot with two panels, one for
    the absolute values of the distributions and one for the
    normalised values. It can also plot the ratio of the distributions
    with respect to a reference distribution.
2. `produce_plot`: Produces a plot with a single panel for the
   distributions.
"""

from pathlib import Path

from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)
rc("text.latex", preamble=r"\usepackage{amsmath,amssymb}")

from yadlt.distribution import Distribution
from yadlt.utils import compute_distance

FONTSIZE = 20
LABELSIZE = 16
LEGENDSIZE = 16
TICKSIZE = 14
FIGSIZE = (10, 5)

PLOT_DIR = Path(__file__).parent.parent / "doc/plots"


def set_plot_dir(plot_dir: str | Path):
    """
    Set the directory where plots will be saved.

    Args:
        plot_dir (str | Path): The directory path to set.
    """
    global PLOT_DIR
    PLOT_DIR = Path(plot_dir)
    if not PLOT_DIR.exists():
        PLOT_DIR.mkdir(parents=True, exist_ok=True)


def get_plot_dir() -> Path:
    """
    Get the directory where plots are saved.

    Returns:
        Path: The directory path where plots are saved.
    """
    return PLOT_DIR


def produce_pdf_plot(
    x_grid,
    grids: list[Distribution],
    normalize_to=1,
    ylabel="",
    xlabel="",
    ratio_label="",
    filename="pdf_plot.pdf",
    title=None,
    ax_specs: list[dict] = [],
    save_fig=True,
    labels: list[str] | None = None,
    colors: list[str] | None = None,
):
    if normalize_to > 0:
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
    for idx, grid in enumerate(grids):
        label = labels[idx] if labels else rf"$\textrm{{{grid.name}}}$"
        color = colors[idx] if colors else None
        pl = axs[0].plot(x_grid, grid.get_mean(), label=label, color=color)
        axs[0].fill_between(
            x_grid,
            grid.get_mean() - grid.get_std(),
            grid.get_mean() + grid.get_std(),
            alpha=0.3,
            color=pl[0].get_color(),
        )
        if normalize_to == 0:
            ref_grid = grid

        normalised_grid = grid / ref_grid.get_mean()
        axs[1].plot(x_grid, normalised_grid.get_mean(), color=pl[0].get_color())
        axs[1].fill_between(
            x_grid,
            normalised_grid.get_mean() - normalised_grid.get_std(),
            normalised_grid.get_mean() + normalised_grid.get_std(),
            alpha=0.3,
            color=pl[0].get_color(),
        )

    # Set the title
    if title is not None:
        fig.suptitle(title, fontsize=FONTSIZE)

    # Set the ratio label
    axs[1].set_ylabel(ratio_label, fontsize=FONTSIZE)

    # X labels
    axs[1].set_xlabel(xlabel, fontsize=FONTSIZE)

    # Y label
    axs[0].set_ylabel(ylabel, fontsize=FONTSIZE)

    # Tick size
    axs[0].tick_params(labelsize=TICKSIZE)
    axs[1].tick_params(labelsize=TICKSIZE)
    # Legend
    axs[0].legend(fontsize=LEGENDSIZE)

    for idx, ax_spec in enumerate(ax_specs):
        ax = axs.flatten()[idx]
        for key, value in ax_spec.items():
            if hasattr(ax, key):
                if isinstance(value, dict):
                    # If value is a dict, assume it's for a method call
                    getattr(ax, key)(**value)
            else:
                getattr(ax, key)(value)
        else:
            print(f"Warning: {key} is not a valid attribute of the Axes object.")

    fig.tight_layout()
    if save_fig:
        fig.savefig(PLOT_DIR / filename, bbox_inches="tight")
    else:
        plt.show()


def produce_plot(
    xgrid,
    grids: list[Distribution],
    additional_grids: list[dict] | None = None,
    ylabel="",
    xlabel="",
    filename="plot.pdf",
    title=None,
    scale="linear",
    ax_specs: dict = {},
    save_fig=True,
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    group_size: int = 1,
):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    label_seen = set()
    for idx, grid in enumerate(grids):
        group_idx = idx // group_size
        color = colors[group_idx] if colors else None
        label = labels[group_idx] if labels else grid.name

        if label in label_seen:
            label = None

        label_seen.add(label)

        pl = ax.plot(xgrid, grid.get_mean(), label=label, color=color)
        ax.fill_between(
            xgrid,
            grid.get_mean() - grid.get_std(),
            grid.get_mean() + grid.get_std(),
            alpha=0.3,
            color=pl[0].get_color(),
        )

    if additional_grids is not None:
        for additional_grid in additional_grids:
            ax.plot(
                xgrid,
                additional_grid["mean"],
                label=additional_grid.get("label", ""),
                color=additional_grid.get("color", None),
            )

    # Set the title
    if title is not None:
        fig.suptitle(title, fontsize=FONTSIZE)

    # Y labels
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    # X labels
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    # Set the x scale
    ax.set_xscale(scale)
    # Tick size
    ax.tick_params(labelsize=TICKSIZE)
    # Legend
    ax.legend(fontsize=LEGENDSIZE)

    for key, value in ax_specs.items():
        if hasattr(ax, key):
            if isinstance(value, dict):
                # If value is a dict, assume it's for a method call
                getattr(ax, key)(**value)
            else:
                getattr(ax, key)(value)
        else:
            print(f"Warning: {key} is not a valid attribute of the Axes object.")

    # Save the figure
    fig.tight_layout()
    if save_fig:
        fig.savefig(PLOT_DIR / filename, bbox_inches="tight")
    else:
        plt.show()


def produce_distance_plot(
    x_grid,
    grids: list[Distribution],
    normalize_to=1,
    filename="distance_plot.pdf",
    title=None,
    ylabel="",
    show_std=False,
    **kwargs,
):
    # Compute distances
    distances = compute_distance(grids, normalize_to=normalize_to)

    if kwargs.get("figsize", None) is None:
        kwargs["figsize"] = FIGSIZE

    # Compute the std
    std = np.sqrt(grids[0].size)

    fig, ax = plt.subplots(**kwargs)
    min_dist, max_dist = 0, 0
    for idx, distance_spec in enumerate(distances):
        name = distance_spec[0]
        distance = distance_spec[1]
        ax.plot(x_grid, distance, label=name)
        if distance.min() < min_dist:
            min_dist = distance.min()
        if distance.max() > max_dist:
            max_dist = distance.max()

    # Add a band for the standard deviation
    if show_std:
        ax.fill_between(
            x_grid,
            -std,
            std,
            alpha=0.3,
            color="gray",
            label=r"$\textrm{Standard deviation}$",
        )
        ax.plot(
            x_grid,
            np.zeros_like(x_grid),
            color="black",
            linestyle="--",
        )

        ymin = max(-std, min_dist) * (1 - 0.1)
        if ymin == 0:
            ymin -= 1

        ymax = max(std, max_dist) * (1 + 0.1)
        ax.set_ylim(ymin, ymax)

    # Set the title
    if title is not None:
        fig.suptitle(title, fontsize=FONTSIZE)
    # Y labels
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    # X labels
    ax.set_xlabel(r"$x$", fontsize=FONTSIZE)
    # Tick size
    ax.tick_params(labelsize=TICKSIZE)
    # Legend
    ax.legend(fontsize=LEGENDSIZE)
    # Save the figure
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, bbox_inches="tight")
