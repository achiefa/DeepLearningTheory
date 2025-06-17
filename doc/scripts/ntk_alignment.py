#!/usr/bin/env python3
"""This script produces a comparison of the following quantities:
1. Eigenvalues of the NTK
2. Relative eigenvalues of the NTK
3. Delta NTK
4. Relative delta NTK
"""
import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)

from argparse import ArgumentParser
import importlib.resources as pkg_resources
from pathlib import Path
import pickle

from yadlt import data
from yadlt.distribution import Distribution

PLOT_DIR = Path(__file__).parent / "../plots"
FIT_PATH = Path(__file__).parent / "../../Results/fits"

FONTSIZE = 20
LABELSIZE = 16
LEGENDSIZE = 16
TICKSIZE = 14


def load_serialized_data(fit_name, data_name, fit_path=FIT_PATH):
    serialization_folder = fit_path / fit_name / "serialization"
    data = pickle.load(open(serialization_folder / f"{data_name}.pickle", "rb"))
    return data


REF_EPOCHS = [0, 4000, 30000, 50000]
REF_REPLICA = 34  # Random replica
REF_FIT = "250604-ac-03-L2"

# Load Tommaso's file
data_path = Path(pkg_resources.files(data) / "BCDMS_data")
fk_grid = np.load(data_path / "fk_grid.npy")
FK = np.load(data_path / "FK.npy")
f_bcdms = np.load(data_path / "f_bcdms.npy")
Cy = np.load(data_path / "Cy.npy")
Cinv = np.linalg.inv(Cy)

# Compute M
M = FK.T @ Cinv @ FK


def produce_mat_plot(args):
    """Produce a comparison of delta NTK for different fits."""

    eigvecs_time = load_serialized_data(
        args.fit, "eigvecs_time", fit_path=Path(args.fit_folder)
    )
    common_epochs = load_serialized_data(
        args.fit, "common_epochs", fit_path=Path(args.fit_folder)
    )
    cut_by_epoch = load_serialized_data(args.fit, "cut", fit_path=Path(args.fit_folder))

    # Compute eigvals and eigvecs of M
    m, W = np.linalg.eigh(M)
    m = m[::-1]
    W = W[:, ::-1]
    Z = eigvecs_time[common_epochs.index(args.epochs[0])]

    for ref_epoch in args.epochs:
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(5, 5),
            gridspec_kw={"left": 0.15, "right": 0.90, "top": 0.99, "bottom": 0.05},
        )
        Z = eigvecs_time[common_epochs.index(ref_epoch)]
        cut = cut_by_epoch[common_epochs.index(ref_epoch)]
        A = np.power(Z[args.replica].T @ W, 2)

        ms = ax.matshow(
            A,
            cmap=mpl.colormaps["RdBu_r"],
            vmax=A.max(),
            vmin=0.0,
            # norm=mpl.colors.LogNorm(vmin=1.e-7, vmax=A.max()),
        )

        # Plot horizontal and vertical lines at the cut value
        cut_value = cut[args.replica]
        ax.axhline(y=cut_value, color="white", linestyle="--", linewidth=2)

        cbar = plt.colorbar(ms, ax=ax, fraction=0.046, pad=0.04)
        # cbar.ax.tick_params(labelsize=TICKSIZE)  # Apply the same tick size as your main axes

        ax.set_title(
            r"$\textrm{Overlap at epoch = }" + f"{ref_epoch}" + r"$", fontsize=FONTSIZE
        )
        ax.set_ylabel(r"$\textrm{Eigenvectors of the NTK}$", fontsize=LABELSIZE)
        ax.set_xlabel(r"$\textrm{Eigenvectors of the M matrix}$", fontsize=LABELSIZE)
        ax.tick_params(labelsize=TICKSIZE)
        ax.xaxis.set_ticks_position("bottom")
        ax.xaxis.set_label_position("bottom")

        # fig.tight_layout()
        fig.savefig(PLOT_DIR / f"overlap_epoch_{ref_epoch}.pdf", dpi=300)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--fit_folder",
        "-f",
        type=str,
        default=FIT_PATH.absolute(),
        help="Path to the folder containing fit results",
    )
    parser.add_argument(
        "--fit",
        type=str,
        default=REF_FIT,
        help="Fit name to use for the comparison",
    )
    parser.add_argument(
        "--replica",
        "-r",
        type=int,
        default=REF_REPLICA,
        help="Replica number to use for the comparison",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        nargs="+",
        default=REF_EPOCHS,
        help="Epochs to use for the comparison",
    )
    args = parser.parse_args()

    produce_mat_plot(args)


if __name__ == "__main__":
    main()
