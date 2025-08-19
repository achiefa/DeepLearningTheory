"""Module to produce plots of the alignment matrix between NTK and M matrices."""

from typing import List

import matplotlib as mpl
from matplotlib import rc
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)

from yadlt.context import FitContext
from yadlt.distribution import combine_distributions
from yadlt.plotting.plotting import FONTSIZE, LABELSIZE, TICKSIZE, produce_plot


def plot_theta_m_alignment(
    context: FitContext,
    replica: int,
    epochs: list[int],
    filename: str,
    save_fig: bool = False,
    plot_dir: str = None,
) -> None:
    """Plot the aligment matrix between the NTK and M matrix for three reference epochs."""
    eigvecs_time = context.eigvecs_time
    cut_by_epoch = context.cut_by_epoch
    common_epochs = context.get_config("replicas", "common_epochs")

    # Compute eigvals and eigvecs of M
    m, W = np.linalg.eigh(context.get_M())
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

    if save_fig:
        fig.savefig(plot_dir / filename, dpi=300)
    else:
        plt.show()


def produce_alignment_plot(
    context: FitContext, eig_ids: List[int] = [0], **plot_kwargs
):
    """Produce a comparison of the alignment between the eigenvectors of the NTK
    and the target input PDF at different epochs.

    Args:
        context (FitContext): The context containing the fit information.
        eig_ids (List[int]): List of eigenvalue indices to plot.
        **plot_kwargs: Additional keyword arguments for the plot.
    """
    # Load eigvecs_time and common_epochs from the context

    eigvecs_time = context.eigvecs_time
    common_epochs = context.get_config("replicas", "common_epochs")

    # Retrieve the input PDF
    fin = context.load_f_bcdms()
    cos_theta_by_time = []

    # Compute cos of the angle between the eigenvectors of the NTK and the input PDF
    for epoch in common_epochs:
        Z = eigvecs_time[common_epochs.index(epoch)]
        fin_normalised = fin / np.linalg.norm(fin, axis=0)
        cos_theta = Z.transpose() @ fin_normalised
        cos_theta_by_time.append(cos_theta)

    cos_theta_combined = combine_distributions(cos_theta_by_time)

    grids = []
    for idx_eig in eig_ids:
        sliced_distribution = cos_theta_combined.slice((slice(None), idx_eig))
        sliced_distribution.set_name(rf"$k = {idx_eig + 1}$")
        grids.append(sliced_distribution)

    # Create auxiliary grid for horizontal line at zero
    add_grid_dict = {
        "mean": np.zeros_like(common_epochs),
        "spec": {
            "linestyle": "--",
            "color": "black",
        },
    }

    # Produce the plot
    produce_plot(
        common_epochs,
        grids,
        scale="linear",
        xlabel=r"${\rm Epoch}$",
        ylabel=r"$\cos \theta^{(k)}$",
        additional_grids=[add_grid_dict],
        **plot_kwargs,
    )
