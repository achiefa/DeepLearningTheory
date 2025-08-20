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

import matplotlib as mpl
from matplotlib import rc
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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


def yield_colors():
    prop_settings = mpl.rcParams["axes.prop_cycle"]
    settings_cycler = prop_settings()

    for color in settings_cycler:
        yield color


next_color = yield_colors()


def produce_pdf_plot(
    x_grid,
    grids: list[Distribution],
    additional_grids: list[dict] | None = None,
    normalize_to=1,
    ylabel="",
    xlabel="",
    ratio_label="",
    filename="pdf_plot.pdf",
    title=None,
    ax_specs: list[dict] = [],
    save_fig=False,
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    plot_dir: Path | None = None,
    legend_title: str = None,
):
    if normalize_to > 0:
        ref_grid = grids[normalize_to - 1].get_mean()
    elif (
        normalize_to < 0 and additional_grids is not None
    ):  # Use additional grids for normalization
        ref_grid = additional_grids[-normalize_to - 1]["mean"]

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
            ref_grid = grid.mean()

        normalised_grid = grid / ref_grid
        axs[1].plot(x_grid, normalised_grid.get_mean(), color=pl[0].get_color())
        axs[1].fill_between(
            x_grid,
            normalised_grid.get_mean() - normalised_grid.get_std(),
            normalised_grid.get_mean() + normalised_grid.get_std(),
            alpha=0.3,
            color=pl[0].get_color(),
        )

    if additional_grids is not None:
        for additional_grid in additional_grids:
            spec = additional_grid.get("spec", {})
            mean = additional_grid.get("mean")
            axs[0].plot(x_grid, mean, **spec)
            axs[1].plot(x_grid, mean / ref_grid, **spec)

    # Set the title
    if title is not None:
        fig.suptitle(title, fontsize=FONTSIZE)

    # Set the ratio label
    axs[1].set_ylabel(ratio_label, fontsize=FONTSIZE)
    axs[1].set_ylim

    # X labels
    axs[1].set_xlabel(xlabel, fontsize=FONTSIZE)

    # Y label
    axs[0].set_ylabel(ylabel, fontsize=FONTSIZE)

    # Tick size
    axs[0].tick_params(labelsize=TICKSIZE)
    axs[1].tick_params(labelsize=TICKSIZE)

    # Legend
    axs[0].legend(
        fontsize=LEGENDSIZE,
        title=legend_title,
        title_fontproperties={"size": LEGENDSIZE},
    )

    # Or for the top plot
    yticks = axs[0].get_yticks()
    axs[0].set_yticks(yticks[1:])  # Remove last tick of top plot
    axs[0].yaxis.set_major_locator(
        MaxNLocator(nbins=5, prune="lower")
    )  # Limit number of ticks

    for idx, ax_spec in enumerate(ax_specs):
        if ax_spec is None:
            continue
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
        fig.savefig(plot_dir / filename, bbox_inches="tight")
        plt.close(fig)
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
    save_fig=False,
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    group_size: int = 1,
    plot_dir: Path | None = None,
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
            spec = additional_grid.get("spec", {})
            ax.plot(xgrid, additional_grid["mean"], **spec)

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
        fig.savefig(plot_dir / filename, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def produce_errorbar_plot(
    xgrid,
    grids: list[Distribution] | None = None,
    add_grids: list[dict] | None = None,
    ylabel: str = "",
    xlabel: str = "",
    title=None,
    scale="linear",
    yscale="linear",
    ax_specs: dict = {},
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    filename: str = "plot.pdf",
    plot_dir: Path | None = None,
    save_fig=False,
):
    """Produce a plot with error bars for the given grids. Grids are meant to be instances of Distribution.
    In alternative, you can pass a list of dictionaries with 'mean' and 'std' keys if instances of Distribution
    are not available. A dictionary in `add_grids` can also contain a 'spec' key for additional plotting specifications,
    so for instance
    ```python
    dict_for_additional_grid = {
        "mean": np.array([1, 2, 3]),
        "std": np.array([0.1, 0.2, 0.3]),
        "label": "Additional Grid",
        "spec": {"color": "red", "linestyle": "--", "label": "Additional Grid"}
    }
    ```
    will plot the additional grid with the specified color and linestyle.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)

    if grids is None and add_grids is None:
        raise ValueError("At least one of 'grids' or 'add_grids' must be provided.")

    legend_handles = []
    legend_labels = []

    if grids is not None:
        for idx, grid in enumerate(grids):
            color = colors[idx] if colors else None
            label = labels[idx] if labels else grid.name

            pl = ax.errorbar(
                x=xgrid,
                y=grid.get_mean(),
                yerr=grid.get_std(),
                label=label,
                color=color,
            )
            legend_handles.append(pl)
            legend_labels.append(label)

    if add_grids is not None:
        for additional_grid in add_grids:
            spec = additional_grid.get("spec", {})
            label = additional_grid["label"]

            tmp_legend_handles = []
            if (
                additional_grid.get("mean") is not None
                and additional_grid.get("std") is not None
            ):
                eb = ax.errorbar(
                    x=xgrid,
                    y=additional_grid["mean"],
                    yerr=additional_grid["std"],
                    **spec,
                )
                tmp_legend_handles.append(eb)

            if collection := additional_grid.get("box"):
                ax.add_collection(collection)
                box_proxy = Rectangle(
                    (0, 0),
                    0.1,
                    0.1,
                    angle=90,
                    facecolor=collection.get_facecolors()[0],
                    alpha=collection.get_alpha(),
                    edgecolor=(
                        collection.get_edgecolors()[0]
                        if len(collection.get_edgecolors()) > 0
                        else "none"
                    ),
                )

                tmp_legend_handles.append(box_proxy)

            legend_handles.append(tuple(tmp_legend_handles))
            legend_labels.append(label)

    # Set the title
    if title is not None:
        fig.suptitle(title, fontsize=FONTSIZE)

    # Y labels
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    # X labels
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    # Set the x and y scale
    ax.set_xscale(scale)
    ax.set_yscale(yscale)
    # Tick size
    ax.tick_params(labelsize=TICKSIZE)
    # Legend
    ax.legend(legend_handles, legend_labels, fontsize=LEGENDSIZE)

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
        fig.savefig(plot_dir / filename, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def produce_distance_plot(
    x_grid,
    grids: list[Distribution],
    normalize_to=1,
    scale="linear",
    title=None,
    ylabel="",
    show_std=False,
    plot_dir: Path | None = None,
    filename="distance_plot.pdf",
    save_fig=False,
    **kwargs,
):
    # Compute distances
    distances = compute_distance(grids, normalize_to=normalize_to)

    if kwargs.get("figsize", None) is None:
        kwargs["figsize"] = FIGSIZE

    # Compute the std
    # Avoid dummy std if the grid size is 1 for the reference grid
    if grids[0].size != 1:
        std = np.sqrt(grids[0].size)
    else:
        std = np.sqrt(grids[1].size)

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
            alpha=0.2,
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
    # Set the x scale
    ax.set_xscale(scale)
    # Save the figure
    fig.tight_layout()
    if save_fig:
        fig.savefig(plot_dir / filename, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def produce_mat_plot(
    matrices: list[np.ndarray],
    titles: list[str],
    text_dict: dict = None,
    vmin: float = None,
    vmax: float = None,
    filename: str = "mat_plot.pdf",
    save_fig: bool = False,
    plot_dir: Path = None,
) -> None:
    """Produce a comparison of delta NTK for different fits."""
    # gridspec_kw = {"left": 0.07, "right": 0.93, "top": 0.99, "bottom": 0.05}

    number_of_matrices = len(matrices)
    vertical = False

    # Check if the first matrix is not square
    if matrices[0].shape[0] != matrices[0].shape[1]:
        vertical = True
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(
            number_of_matrices + 1,
            1,
            height_ratios=[1] * number_of_matrices + [0.2],
            hspace=0.2,
        )
        cax = fig.add_subplot(gs[-1, 0])  # Colorbar axis
        axs = [fig.add_subplot(gs[i, 0]) for i in range(number_of_matrices)]
    else:
        fig = plt.figure(figsize=(12, 5))
        gs = GridSpec(
            1,
            number_of_matrices + 1,
            width_ratios=[1] * number_of_matrices + [0.08],
            wspace=0.2,
            left=0.15 if text_dict else 0.1,
        )
        cax = fig.add_subplot(gs[0, -1])  # Colorbar axis
        axs = [fig.add_subplot(gs[0, i]) for i in range(number_of_matrices)]
        if text_dict is not None:
            fig.text(
                **text_dict,
                ha="center",
                va="center",
                transform=axs[0].transAxes,
                fontsize=FONTSIZE,
            )

    if vmin is None or vmax is None:
        vmin = min(np.percentile(A, 1) for A in matrices)
        vmax = max(np.percentile(A, 95) for A in matrices)

    for idx, ax in enumerate(axs):
        matrix = matrices[idx]
        ms = ax.matshow(
            matrix,
            cmap=mpl.colormaps["RdBu_r"],
            norm=Normalize(
                vmin=vmin, vmax=vmax, clip=True
            ),  # clip=True will clip out-of-range values4
        )

        ax.set_title(titles[idx], fontsize=FONTSIZE, pad=10)

        ax.tick_params(labelsize=TICKSIZE)
        ax.xaxis.set_ticks_position("bottom")
        ax.xaxis.set_label_position("bottom")

    cbar = plt.colorbar(
        ms, cax=cax, extend="both", orientation="horizontal" if vertical else "vertical"
    )

    # Get the actual position of one of your matshow plots (they should all be the same height)
    pos = axs[0].get_position()
    cax_pos = cax.get_position()
    if vertical:
        cax.set_position([pos.x0, cax_pos.y0, pos.width, cax_pos.height])
    else:
        cax.set_position([cax_pos.x0, pos.y0, cax_pos.width, pos.height])

    cbar.ax.tick_params(
        labelsize=TICKSIZE
    )  # Apply the same tick size as your main axes
    if save_fig:
        fig.savefig(plot_dir / filename, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
    else:
        plt.show()
