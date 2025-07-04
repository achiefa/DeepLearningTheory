#!/usr/bin/env python3
"""This script produces a comparison of the following quantities:
1. Eigenvalues of the NTK
2. Relative eigenvalues of the NTK
3. Delta NTK
4. Relative delta NTK
"""
import matplotlib as mpl
from matplotlib import rc
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import yaml

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)

from argparse import ArgumentParser

from yadlt.evolution import EvolutionOperatorComputer
from yadlt.plotting import FONTSIZE, LABELSIZE, PLOT_DIR, TICKSIZE


def produce_mat_plot(
    evolution: EvolutionOperatorComputer, replica: int, epochs: list[int], filename: str
):
    """Produce a comparison of delta NTK for different fits."""

    eigvecs_time = evolution.eigvecs_time
    common_epochs = evolution.common_epochs
    cut_by_epoch = evolution.cut_by_epoch

    # Compute eigvals and eigvecs of M
    m, W = np.linalg.eigh(evolution.M)
    m = m[::-1]
    W = W[:, ::-1]

    gridspec_kw = {"left": 0.07, "right": 0.93, "top": 0.99, "bottom": 0.05}
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.07], wspace=0.20, **gridspec_kw)

    # Create axes
    ax1 = fig.add_subplot(gs[0, 0])  # Left plot
    ax2 = fig.add_subplot(gs[0, 1])  # Left plot
    ax3 = fig.add_subplot(gs[0, 2])  # Left plot
    cax = fig.add_subplot(gs[0, 3])  # Colorbar axis
    axs = [ax1, ax2, ax3]

    matrices = []
    cut_values = []

    # Compute the overlap matrices for each reference epoch
    for ref_epoch in epochs:
        Z = eigvecs_time[common_epochs.index(ref_epoch)]
        cut = cut_by_epoch[common_epochs.index(ref_epoch)]
        A = np.power(Z[replica].T @ W, 2)
        cut_value = cut[replica]
        cut_values.append(cut_value)
        matrices.append(A)

    vmin = min(np.percentile(A, 1) for A in matrices)
    vmax = max(np.percentile(A, 95) for A in matrices)
    print(f"vmin: {vmin}, vmax: {vmax}")

    for idx, ax in enumerate(axs):
        ms = ax.matshow(
            matrices[idx],
            cmap=mpl.colormaps["RdBu_r"],
            norm=Normalize(
                vmin=vmin, vmax=vmax, clip=True
            ),  # clip=True will clip out-of-range values4
        )

        # Plot horizontal and vertical lines at the cut value
        ax.axhline(y=cut_values[idx], color="white", linestyle="--", linewidth=2)
        ax.set_title(
            r"$\textrm{Overlap at epoch = }" + f"{epochs[idx]}" + r"$",
            fontsize=FONTSIZE,
        )

        ax.tick_params(labelsize=TICKSIZE)
        ax.xaxis.set_ticks_position("bottom")
        ax.xaxis.set_label_position("bottom")

    cbar = plt.colorbar(ms, cax=cax, extend="both")
    cbar.ax.yaxis.set_label_position("left")

    # Get the actual position of one of your matshow plots (they should all be the same height)
    pos = ax1.get_position()
    cax_pos = cax.get_position()
    cax.set_position([cax_pos.x0, pos.y0, cax_pos.width, pos.height])

    cbar.ax.tick_params(
        labelsize=TICKSIZE
    )  # Apply the same tick size as your main axes

    ax1.set_ylabel(r"$\textrm{Eigenvectors of the NTK}$", fontsize=LABELSIZE)
    ax2.set_xlabel(r"$\textrm{Eigenvectors of the M matrix}$", fontsize=LABELSIZE)

    fig.savefig(PLOT_DIR / filename, dpi=300)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        help="Path to the config file",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Directory to save the plots. If not specified, uses the default plot directory.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="ntk_alignment.pdf",
        help="Filename to save the plot.",
    )
    args = parser.parse_args()

    if args.plot_dir is not None:
        from yadlt.plotting import set_plot_dir

        set_plot_dir(args.plot_dir)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    fitname = config["fitname"]
    replica = config["replica"]
    epochs = config["epochs"]

    evolution = EvolutionOperatorComputer(fitname)
    produce_mat_plot(
        evolution=evolution, replica=replica, epochs=epochs, filename=args.filename
    )


if __name__ == "__main__":
    main()
