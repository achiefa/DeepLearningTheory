"""
This script produces the following plots:
- Comparison of the expectation value of the product U @ f0 with
  with the product of the expectation values of U and f0.
- Comparison of the empirical covariance of ft with the
  "theoretical" covariance computed as U_bar @ Cov[f0, f0] @ U_bar^T.
- Comparison of U_bar Vs Cov[f0, f0].
- Fluctuations of U and V at different epochs.
- Comparison of the Gibbs kernel with Cov[f0, f0]

The plots above can be iterated for different fits.
"""

from argparse import ArgumentParser
from functools import partial

import matplotlib as mpl
from matplotlib import rc
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import yaml

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)

from pdf_plots import SEED, load_data, produce_model_at_initialisation

from yadlt.distribution import Distribution
from yadlt.evolution import EvolutionOperatorComputer
from yadlt.plotting import FONTSIZE, TICKSIZE, get_plot_dir, produce_plot

# Define GP kernel parameters
SIGMA = 0.25
L0 = 1.7
DELTA = 1.0e-9
ALPHA = -0.1


def gibbs_fn(fk_grid, i1, i2, rescl=False):
    """
    Compute the (i1, i2) entry of the Gibbs kernel for the given grid.
    """
    x1 = fk_grid[i1]
    x2 = fk_grid[i2]

    def l(x):
        return L0 * (x + DELTA)

    tmp = (
        SIGMA**2
        * np.sqrt(2 * l(x1) * l(x2) / (np.power(l(x1), 2) + np.power(l(x2), 2)))
        * np.exp(-np.power(x1 - x2, 2) / (np.power(l(x1), 2) + np.power(l(x2), 2)))
    )

    if rescl:
        return np.power(x1, ALPHA) * np.power(x2, ALPHA) * tmp

    return tmp * x1 * x2


def produce_mat_plot(
    matrices: list[np.ndarray], titles: list[str], filename: str, save: bool = True
) -> None:
    """Produce a comparison of delta NTK for different fits."""
    vertical = False
    # gridspec_kw = {"left": 0.07, "right": 0.93, "top": 0.99, "bottom": 0.05}

    if matrices[0].shape[0] != matrices[0].shape[1]:
        vertical = True
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(3, 1, height_ratios=[1, 1, 0.2], hspace=0.2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        cax = fig.add_subplot(gs[2, 0])  # Colorbar axis
        axs = [ax1, ax2]
    else:
        fig = plt.figure(figsize=(12, 5))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 0.08], wspace=0.2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        cax = fig.add_subplot(gs[0, 2])  # Colorbar axis
        axs = [ax1, ax2]

    vmin = min(np.percentile(A, 1) for A in matrices)
    vmax = max(np.percentile(A, 95) for A in matrices)

    print(f"vmin: {vmin}, vmax: {vmax}")

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
    pos = ax1.get_position()
    cax_pos = cax.get_position()
    if vertical:
        cax.set_position([pos.x0, cax_pos.y0, pos.width, cax_pos.height])
    else:
        cax.set_position([cax_pos.x0, pos.y0, cax_pos.width, pos.height])

    cbar.ax.tick_params(
        labelsize=TICKSIZE
    )  # Apply the same tick size as your main axes

    if save:
        fig.savefig(get_plot_dir() / filename, dpi=300)
    else:
        plt.show()


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        help="Config file",
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
        default="delta_ntk.pdf",
        help="Filename to save the plot.",
    )
    args = parser.parse_args()

    if args.plot_dir is not None:
        from yadlt.plotting import set_plot_dir

        set_plot_dir(args.plot_dir)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    fitnames = config["fitnames"]
    reference_epoch = config.get("reference_epoch", None)
    epochs = config.get("epochs", None)
    seed = config.get("seed", SEED)

    # As of now, only two epochs are supported for the comparison
    if len(epochs) != 2:
        raise ValueError(
            "This script only supports two epochs for the comparison "
            f"but {len(epochs)} epochs were provided. Please provide exactly two epochs."
        )

    # Generate model at initialisation
    evolution = EvolutionOperatorComputer(fitnames[0])
    xT3_init = produce_model_at_initialisation(
        evolution.replicas,
        tuple(evolution.fk_grid),
        tuple(evolution.metadata["model_info"]["architecture"]),
        seed=seed,
    )
    f0_mean = xT3_init.get_mean()

    # Compute covariance of f0
    cov_f0 = Distribution(
        name=r"$\textrm{Cov}[f_0, f_0]$",
        size=f0_mean.size,
        shape=(f0_mean.shape[0], f0_mean.shape[0]),
    )
    for f0_rep in xT3_init:
        cov = np.outer(f0_rep, f0_rep)
        cov_f0.add(cov)

    # Produce comparison with Gibbs kernel
    gibbs_fn_partial = partial(gibbs_fn, evolution.fk_grid, rescl=False)
    gp_kernel_rscl = np.fromfunction(
        gibbs_fn_partial, (evolution.fk_grid.size, evolution.fk_grid.size), dtype=int
    )
    produce_mat_plot(
        matrices=[gp_kernel_rscl, cov_f0.get_mean()],
        titles=[r"$\textrm{Gibbs}$", r"$\textrm{Cov}[f_0, f_0]$"],
        save=True,
        filename=f"gibbs_comparison_no_low_x.pdf",
    )

    for fitname in fitnames:
        # Compute evolution operators U and V
        evolution = EvolutionOperatorComputer(fitname)
        learning_rate = evolution.learning_rate
        data_type = evolution.metadata["arguments"]["data"]
        data = load_data(evolution)

        U_fluctuations_by_epochs = []
        V_fluctuations_by_epochs = []

        for epoch in epochs:
            t = epoch * learning_rate
            U, V = evolution.compute_evolution_operator(reference_epoch, t)

            # Compute fluctuations for U and V
            U_mean = U.get_mean()
            U_std = U.get_std()
            U_fluctuations = U_std / np.abs(U_mean)

            V_mean = V.get_mean()
            V_std = V.get_std()
            V_fluctuations = V_std / np.abs(V_mean)
            np.nan_to_num(V_fluctuations, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            U_fluctuations_by_epochs.append(U_fluctuations)
            V_fluctuations_by_epochs.append(V_fluctuations)

            # Compute grid for U f0
            Uf0 = U @ xT3_init
            Uf0.set_name(r"$\mathbb{E}[U f_0]$")

            # Compute full evolution
            ft = Uf0 + V @ data

            # Compute covariance of ft
            cov_ft_rhs = U_mean @ cov_f0.get_mean() @ U_mean.T

            # Compare the mean value of U with that of the covariance of f0
            produce_mat_plot(
                matrices=[U_mean, cov_f0.get_mean()],
                titles=[r"$\bar{U}$", r"$\textrm{Cov}[f_0, f_0]$"],
                save=True,
                filename=f"U_bar_vs_cov_f0_{epoch}_{data_type}.pdf",
            )

            cov_ft = Distribution(
                name=r"$\textrm{Cov}[f_t, f_t]$",
                size=ft.size,
                shape=(ft.shape[0], ft.shape[0]),
            )
            for ft_rep in ft:
                cov = np.outer(ft_rep, ft_rep)
                cov_ft.add(cov)

            # Produce plots of the expectation value of U f0
            add_grid_dict = {
                "mean": U_mean @ f0_mean,
                "label": r"$\mathbb{E}[U] \mathbb{E}[f_0]$",
                "color": "black",
            }

            # Produce plots of the expectation value of U f0
            produce_plot(
                evolution.fk_grid,
                [Uf0],
                additional_grids=[add_grid_dict],
                ylabel="$U f_0$",
                xlabel="$x$",
                title=rf"$T={epoch}$",
                scale="linear",
                save_fig=True,
                filename=f"u_f0_independence_{epoch}_{data_type}.pdf",
            )

            # Produce plots of the matrices of the covariance of ft
            produce_mat_plot(
                matrices=[cov_ft.get_mean(), cov_ft_rhs],
                titles=[
                    r"$\textrm{Cov}[f_t, f_t]$",
                    r"$\mathbb{E}[U] \textrm{Cov}[f_0, f_0] \mathbb{E}[U]^T$",
                ],
                filename=f"cov_ft_{epoch}_{data_type}.pdf",
                save=True,
            )

        # Produce plots of the fluctuations of U and V
        produce_mat_plot(
            matrices=U_fluctuations_by_epochs,
            titles=[rf"$\delta U \textrm{{ at }} T = {epoch}$" for epoch in epochs],
            filename=f"u_fluctuations_{data_type}.pdf",
            save=True,
        )
        produce_mat_plot(
            matrices=V_fluctuations_by_epochs,
            titles=[rf"$\delta V \textrm{{ at }} T = {epoch}$" for epoch in epochs],
            filename=f"v_fluctuations_{data_type}.pdf",
            save=True,
        )


if __name__ == "__main__":
    main()
