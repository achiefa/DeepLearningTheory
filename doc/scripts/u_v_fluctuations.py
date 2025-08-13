"""
This script produces the following plots:
- Comparison of the expectation value of the product U @ f0 with
  with the product of the expectation values of U and f0.
- Comparison of the empirical covariance of ft with the
  "theoretical" covariance computed as U_bar @ Cov[f0, f0] @ U_bar^T.
- Comparison of U_bar Vs Cov[f0, f0].
- Fluctuations of U and V at different epochs.

The plots above can be iterated for different fits.
"""

from argparse import ArgumentParser
import logging
from pathlib import Path

import numpy as np
from pdf_plots import SEED

from yadlt.context import FitContext
from yadlt.distribution import Distribution
from yadlt.evolution import EvolutionOperatorComputer
from yadlt.log import setup_logger
from yadlt.plotting.plotting import produce_mat_plot
from yadlt.utils import load_data, produce_model_at_initialisation

logger = setup_logger()
logger.setLevel(logging.INFO)

# Define GP kernel parameters
SIGMA = 0.25
L0 = 1.7
DELTA = 1.0e-9
ALPHA = -0.1

PLOT_DICT = {
    "fitnames": [
        # Fit names to be used for the comparison
        "250713-01-L0-nnpdf-like",
        "250713-02-L1-nnpdf-like",
    ],
    "reference_epoch": 20000,
    "epochs": [0, 20000],
    "seed": 1423413,
}


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Directory to save the plots. If not specified, uses the default plot directory.",
    )
    args = parser.parse_args()
    DIR = Path(args.plot_dir) if args.plot_dir else Path(__file__).parent / "plots"
    PLOT_DIR = DIR / "u_v_studies"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    fitnames = PLOT_DICT["fitnames"]
    reference_epoch = PLOT_DICT["reference_epoch"]
    epochs = PLOT_DICT["epochs"]
    seed = PLOT_DICT["seed"]

    # As of now, only two epochs are supported for the comparison
    if len(epochs) != 2:
        raise ValueError(
            "This script only supports two epochs for the comparison "
            f"but {len(epochs)} epochs were provided. Please provide exactly two epochs."
        )

    # Generate model at initialisation
    context = FitContext.get_instance(fitnames[0])
    xT3_init = produce_model_at_initialisation(
        context.get_property("nreplicas"),
        tuple(context.load_fk_grid()),
        tuple(context.get_config("metadata", "model_info")["architecture"]),
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

    for fitname in fitnames:
        # Compute evolution operators U and V
        context = FitContext.get_instance(fitname)
        evolution = EvolutionOperatorComputer(context)

        learning_rate = float(
            context.get_config("metadata", "arguments")["learning_rate"]
        )
        data_type = context.get_config("metadata", "arguments")["data"]
        data = load_data(context)

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
                save_fig=True,
                filename=f"U_bar_vs_cov_f0_{epoch}_{data_type}.pdf",
                plot_dir=PLOT_DIR,
            )

            cov_ft = Distribution(
                name=r"$\textrm{Cov}[f_t, f_t]$",
                size=ft.size,
                shape=(ft.shape[0], ft.shape[0]),
            )
            for ft_rep in ft:
                cov = np.outer(ft_rep, ft_rep)
                cov_ft.add(cov)

            # Produce plots of the matrices of the covariance of ft
            produce_mat_plot(
                matrices=[cov_ft.get_mean(), cov_ft_rhs],
                titles=[
                    r"$\textrm{Cov}[f_t, f_t]$",
                    r"$\mathbb{E}[U] \textrm{Cov}[f_0, f_0] \mathbb{E}[U]^T$",
                ],
                filename=f"cov_ft_{epoch}_{data_type}.pdf",
                save_fig=True,
                plot_dir=PLOT_DIR,
            )

        # Produce plots of the fluctuations of U and V
        produce_mat_plot(
            matrices=U_fluctuations_by_epochs,
            titles=[rf"$\delta U \textrm{{ at }} T = {epoch}$" for epoch in epochs],
            filename=f"u_fluctuations_{data_type}.pdf",
            save_fig=True,
            plot_dir=PLOT_DIR,
        )
        produce_mat_plot(
            matrices=V_fluctuations_by_epochs,
            titles=[rf"$\delta V \textrm{{ at }} T = {epoch}$" for epoch in epochs],
            filename=f"v_fluctuations_{data_type}.pdf",
            save_fig=True,
            plot_dir=PLOT_DIR,
        )


if __name__ == "__main__":
    main()
