#!/usr/bin/env python3
"""This script produces the the comparisons of the NTK at initialization"""

from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)

from argparse import ArgumentParser
import importlib.resources as pkg_resources
from pathlib import Path

import tensorflow as tf

from yadlt import data
from yadlt.distribution import Distribution
from yadlt.model import PDFmodel, compute_ntk_static

PLOT_DIR = Path(__file__).parent / "../plots"
FIT_PATH = Path(__file__).parent / "../../Results/fits"

# Load Tommaso's file
data_path = Path(pkg_resources.files(data) / "BCDMS_data")
fk_grid = np.load(data_path / "fk_grid.npy")

SEED = 2312441
NREPLICAS = 100


def compute_ntk_initialisation(n_rep, seed, architecture=[28, 20]):
    ntk_by_reps = Distribution("NTK by replicas", shape=(), size=n_rep)
    eigvals_by_reps = Distribution(
        "Eigenvalues by replicas", shape=(fk_grid.shape[0],), size=n_rep
    )

    for rep in range(n_rep):
        model = PDFmodel(
            dense_layer="Dense",
            input=fk_grid,
            outputs=1,
            architecture=architecture,
            activations=["tanh" for _ in architecture],
            kernel_initializer="GlorotNormal",
            user_ki_args=None,
            seed=seed + rep,
        )
        _ = model.model(tf.convert_to_tensor(fk_grid.reshape(-1, 1)))
        ntk = compute_ntk_static(
            tf.convert_to_tensor(fk_grid.reshape(-1, 1)), model.model, model.outputs
        )
        ntk_by_reps.add(np.linalg.norm(ntk.numpy()))

        eigvals = np.linalg.eigvalsh(ntk.numpy())
        eigvals = np.sort(eigvals)[::-1]  # Sort in descending order
        eigvals_by_reps.add(eigvals)

    mean = ntk_by_reps.get_mean()
    std = ntk_by_reps.get_std()

    eigvals_mean = eigvals_by_reps.get_mean(axis=0)
    eigvals_std = eigvals_by_reps.get_std(axis=0)

    return (mean, std, eigvals_mean, eigvals_std)


ARCHITECTURES = [
    [10, 10],
    [28, 20],
    [100, 100],
    [1000, 1000],
]


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
        "--n_replicas",
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
    args = parser.parse_args()

    # Lists to store results
    means = []
    stds = []
    eigvals_means = []
    eigvals_stds = []

    # Compute means and standard deviations for each architecture
    for arch in ARCHITECTURES:
        print(f"Computing NTK for architecture {arch}")
        mean, std, eigvals_mean, eigvals_std = compute_ntk_initialisation(
            args.n_replicas, args.seed, architecture=arch
        )
        means.append(mean)
        stds.append(std)
        eigvals_means.append(eigvals_mean)
        eigvals_stds.append(eigvals_std)

    # ========== Plot of Frobenius Norm of NTK at Initialization ==========
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Calculate relative uncertainties (coefficient of variation)
    rel_uncertainties = [s / m for s, m in zip(stds, means)]
    # X-axis labels for architectures
    x_labels = [rf"$[{a[0]},{a[1]}]$" for a in ARCHITECTURES]
    x_positions = np.arange(len(ARCHITECTURES))

    # NTK Norm
    ax1.errorbar(
        x_positions,
        means,
        yerr=stds,
        fmt="o",
        capsize=5,
        markersize=8,
        elinewidth=2,
        label="NTK Norm",
    )

    # Add a subtle shaded area for the error bands
    for i, (m, s) in enumerate(zip(means, stds)):
        ax1.fill_between([i - 0.1, i + 0.1], [m - s, m - s], [m + s, m + s], alpha=0.2)

    ax1.set_xlabel(r"${\rm Architecture}$", fontsize=14)
    ax1.set_ylabel(r"$\textrm{NTK Norm}$", fontsize=14)
    ax1.set_title(r"$\textrm{Neural Tangent Kernel Norm}$", fontsize=16)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_labels, rotation=45)
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Uncertainty Trend
    ax2.plot(
        x_positions, rel_uncertainties, "o-", markersize=8, linewidth=2, color="green"
    )
    ax2.plot()
    ax2.set_xlabel(r"${\rm Architecture}$", fontsize=14)
    ax2.set_ylabel(r"$\textrm{Relative Uncertainty}$", fontsize=14)
    ax2.set_title(r"$\textrm{Uncertainty vs. Architecture Size}$", fontsize=16)
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(x_labels, rotation=45)
    ax2.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    # Save the combined plot
    output_path = PLOT_DIR / "ntk_initialization_with_uncertainty.pdf"
    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()

    # ========== Plot of Eigenvalues of NTK at Initialization ==========
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))

    # Colors for different architectures
    colors = ["blue", "red", "green", "purple", "orange"]

    # Compute and plot eigenvalues for each architecture
    for i, arch in enumerate(ARCHITECTURES):
        print(f"Computing NTK eigenvalues for architecture {arch}")

        # Get means and standard deviations
        means = eigvals_means[i]
        stds = eigvals_stds[i]

        # Plot only the first 30 eigenvalues (or adjust as needed)
        max_eigvals = min(30, len(means))
        x = np.arange(max_eigvals)

        ax.errorbar(
            x,
            means[:max_eigvals],
            yerr=stds[:max_eigvals],
            fmt="o-",
            capsize=3,
            markersize=5,
            color=colors[i % len(colors)],
            label=f"Architecture {arch}",
        )

    # Use log scale for y-axis
    ax.set_yscale("log")

    # Customize plot
    ax.set_xlabel(r"$\textrm{Eigenvalue Index}$", fontsize=14)
    ax.set_ylabel(r"$\textrm{Eigenvalue Magnitude}$", fontsize=14)
    ax.set_title(r"$\textrm{NTK Eigenvalue Spectrum by Architecture}$", fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()

    # Save and show plot
    output_path = PLOT_DIR / "ntk_eigenvalue_spectrum.pdf"
    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path)
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
