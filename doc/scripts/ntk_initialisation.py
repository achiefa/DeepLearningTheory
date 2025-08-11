#!/usr/bin/env python3
"""This script produces the the comparisons of the NTK at initialization"""

import logging

from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)

from argparse import ArgumentParser
from pathlib import Path
import pickle

from yadlt import load_data
from yadlt.distribution import Distribution
from yadlt.log import setup_logger
from yadlt.model import compute_ntk_static, generate_pdf_model
from yadlt.plotting.plotting import FONTSIZE, LABELSIZE, LEGENDSIZE, TICKSIZE

logger = setup_logger()
logger.setLevel(logging.INFO)

SERIALIZATION_FOLDER = Path(__file__).parent / "serialization"

# Load Tommaso's file
fk_grid = load_data.load_bcdms_grid()
x = fk_grid.reshape(1, -1, 1)

SEED = 2312441
NREPLICAS = 100


def compute_ntk_initialisation(n_rep, seed, architecture=[28, 20]):
    ntk_by_reps = Distribution(
        "NTK byreplicas", shape=(fk_grid.shape[0], fk_grid.shape[0]), size=n_rep
    )
    eigvals_by_reps = Distribution(
        "Eigenvalues by replicas", shape=(fk_grid.shape[0],), size=n_rep
    )
    frob_norm_by_reps = Distribution("Frobenius Norm by replicas", shape=(), size=n_rep)

    for rep in range(n_rep):
        model = generate_pdf_model(
            outputs=1,
            architecture=architecture,
            activations=["tanh" for _ in architecture],
            kernel_initializer="GlorotNormal",
            user_ki_args=None,
            seed=seed + rep,
            scaled_input=False,
            preprocessing=False,
        )
        ntk = compute_ntk_static(x, model, 1)
        ntk_by_reps.add(ntk.numpy())

        frob_norm = np.linalg.norm(ntk.numpy(), ord="fro")
        frob_norm_by_reps.add(frob_norm)

        eigvals = np.linalg.eigvalsh(ntk.numpy())
        eigvals = np.sort(eigvals)[::-1]  # Sort in descending order
        eigvals_by_reps.add(eigvals)

    return (ntk_by_reps, eigvals_by_reps, frob_norm_by_reps)


ARCHITECTURES = [
    [10, 10],
    [25, 20],
    [100, 100],
    [1000, 1000],
    # [2000, 2000],
    # [10000, 10000],
]


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--n-replicas",
        "-n",
        type=int,
        default=NREPLICAS,
        help="Number of replicas to compute the NTK",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=SEED,
        help="Seed for the random number generator",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Directory to save the plots. If not specified, uses the default plot directory.",
    )
    args = parser.parse_args()
    DIR = Path(args.plot_dir) if args.plot_dir else Path(__file__).parent / "plots"
    PLOT_DIR = DIR / "ntk_pheno"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Lists to store results
    group_dict = {}

    # Compute means and standard deviations for each architecture
    for arch in ARCHITECTURES:
        if not (
            serialization_path := SERIALIZATION_FOLDER
            / f"ntk_initialization_{arch[0]}.pkl"
        ).exists():
            print(f"Computing NTK for architecture {arch}")
            ntk_by_reps, eigvals_by_reps, frob_norm_by_reps = (
                compute_ntk_initialisation(
                    args.n_replicas, args.seed, architecture=arch
                )
            )
            pickle.dump(
                ntk_by_reps,
                open(SERIALIZATION_FOLDER / f"ntk_initialization_{arch[0]}.pkl", "wb"),
            )
            pickle.dump(
                eigvals_by_reps,
                open(
                    SERIALIZATION_FOLDER / f"eigvals_initialization_{arch[0]}.pkl", "wb"
                ),
            )
            pickle.dump(
                frob_norm_by_reps,
                open(
                    SERIALIZATION_FOLDER / f"frob_norm_initialization_{arch[0]}.pkl",
                    "wb",
                ),
            )
            print(f"Results saved to {serialization_path}")
        else:
            print(f"Loading NTK for architecture {arch} from {serialization_path}")
            ntk_by_reps = pickle.load(
                open(SERIALIZATION_FOLDER / f"ntk_initialization_{arch[0]}.pkl", "rb")
            )
            eigvals_by_reps = pickle.load(
                open(
                    SERIALIZATION_FOLDER / f"eigvals_initialization_{arch[0]}.pkl", "rb"
                )
            )
            frob_norm_by_reps = pickle.load(
                open(
                    SERIALIZATION_FOLDER / f"frob_norm_initialization_{arch[0]}.pkl",
                    "rb",
                )
            )

        group_dict[tuple(arch)] = {
            "ntk": ntk_by_reps,
            "eigvals": eigvals_by_reps,
            "frob_norm": frob_norm_by_reps,
        }

    # ========== Plot of Frobenius Norm of NTK at Initialization ==========
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Calculate relative uncertainties (coefficient of variation)
    # rel_uncertainties = [s / m for s, m in zip(stds, means)]
    mean_values = [arch["frob_norm"].get_mean() for arch in group_dict.values()]
    uncertainties = [arch["frob_norm"].get_std() for arch in group_dict.values()]
    rel_uncertainties = [
        arch["frob_norm"].get_std() / np.abs(arch["frob_norm"].get_mean())
        for arch in group_dict.values()
    ]

    # X-axis labels for architectures
    x_labels_ax1 = [rf"$[{a[0]},{a[1]}]$" for a in ARCHITECTURES]
    x_positions_ax1 = np.arange(len(ARCHITECTURES))

    # NTK Norm
    ax1.errorbar(
        x_positions_ax1,
        mean_values,
        yerr=uncertainties,
        fmt="o",
        capsize=5,
        markersize=8,
        elinewidth=2,
        label="NTK Norm",
        color="C0",
    )

    # Add a subtle shaded area for the error bands
    # for i, (m, s) in enumerate(zip(mean_values, uncertainties)):
    #     ax1.fill_between([i - 0.1, i + 0.1], [m - s, m - s], [m + s, m + s], alpha=0.2)

    ax1.set_xlabel(r"${\rm Architecture}$", fontsize=LABELSIZE)
    ax1.set_ylabel(r"$\textrm{NTK Norm}$", fontsize=LABELSIZE)
    ax1.set_title(r"$\textrm{Neural Tangent Kernel Norm}$", fontsize=FONTSIZE)
    ax1.set_xticks(x_positions_ax1)
    ax1.set_xticklabels(x_labels_ax1, rotation=45, fontsize=TICKSIZE)
    # ax1.grid(True, linestyle="--", alpha=0.7)

    # Uncertainty Trend
    x_positions_ax2 = [a[0] for a in ARCHITECTURES]
    ax2.plot(
        x_positions_ax2, rel_uncertainties, "o-", markersize=8, linewidth=2, color="C0"
    )
    ax2.plot()
    ax2.set_xlabel(r"${\rm Architecture}$", fontsize=LABELSIZE)
    ax2.set_ylabel(r"$\textrm{Relative Uncertainty}$", fontsize=LABELSIZE)
    ax2.set_title(r"$\textrm{Uncertainty vs. Architecture Size}$", fontsize=FONTSIZE)
    ax2.set_xticks(x_positions_ax2)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    fig.tight_layout()

    # Save the combined plot
    output_path = PLOT_DIR / "ntk_initialization_with_uncertainty.pdf"
    fig.savefig(output_path)
    print(f"Plot saved to {output_path}")

    # ========== Plot of Eigenvalues of NTK at Initialization ==========
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # # Colors for different architectures
    # colors = ["blue", "red", "green", "purple", "orange"]

    # Compute and plot eigenvalues for each architecture
    mean_eigvals_by_arch = []
    std_eigvals_by_arch = []
    for arch in group_dict.values():
        eigvals = arch["eigvals"]
        tmp_mean = [mean for mean in eigvals.get_mean()[:5]]
        tmp_std = [std for std in eigvals.get_std()[:5]]
        mean_eigvals_by_arch.append(tmp_mean)
        std_eigvals_by_arch.append(tmp_std)

    mean_eigvals_by_arch = np.array(mean_eigvals_by_arch)  # [arch, eigvals]
    std_eigvals_by_arch = np.array(std_eigvals_by_arch)  # [arch, eigvals]

    # Store handles for legend
    handles = []
    labels = []

    for idx in range(mean_eigvals_by_arch.shape[1]):
        means = mean_eigvals_by_arch[:, idx]
        stds = std_eigvals_by_arch[:, idx]

        # Plot the eigenvalues with error bars
        line = ax.errorbar(
            x_positions_ax1,
            means,
            yerr=stds,
            fmt="o",
            capsize=2,
            markersize=4,
            label=rf"$\lambda^{{{idx + 1}}}$",
            color=f"C{idx}",
        )

        # Plot horizontal line for last architecture
        ax.axhline(
            y=mean_eigvals_by_arch[-1, idx],
            color=line[0].get_color(),
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )

        handles.append(line)
        labels.append(rf"$\lambda^{{{idx + 1}}}$")

        for i, (m, s) in enumerate(zip(means, stds)):
            ax.fill_between(
                [i - 0.1, i + 0.1],
                [m - s, m - s],
                [m + s, m + s],
                alpha=0.2,
                color=f"C{idx}",
            )

        ax.set_xlabel(r"${\rm Architecture}$", fontsize=LABELSIZE)
        ax.set_ylabel(r"$\textrm{NTK eigenvalues}$", fontsize=LABELSIZE)
        ax.set_title(r"$\textrm{NTK with different architectures}$", fontsize=FONTSIZE)
        ax.set_xticks(x_positions_ax1)
        ax.set_xticklabels(x_labels_ax1, rotation=45, fontsize=TICKSIZE)
        ax.set_yscale("log")

    ax.legend(handles, labels, fontsize=LEGENDSIZE)

    # Save the combined plot
    output_path = PLOT_DIR / "ntk_initialization_arch.pdf"
    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
