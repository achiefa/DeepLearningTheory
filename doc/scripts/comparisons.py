#!/usr/bin/env python3
"""This script produces a comparison of the following quantities:
1. Eigenvalues of the NTK
2. Relative eigenvalues of the NTK
3. Delta NTK
4. Relative delta NTK
"""
from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)

from argparse import ArgumentParser
from pathlib import Path
import pickle

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


COMPARISON_CONFIGS = {
    "data_generation": {
        "fits": [
            "250604-ac-01-L0",
            "250604-ac-02-L1",
            "250604-ac-03-L2",
        ],
        "plot_args": [
            {"color": "blue", "marker": "o", "label": r"$\textrm{L0}$"},
            {"color": "orange", "marker": "s", "label": r"$\textrm{L1}$"},
            {"color": "green", "marker": "D", "label": r"$\textrm{L2}$"},
        ],
        "suffix": "",
    },
    "architecture": {
        "fits": {
            "L0": ["250604-ac-01-L0", "250605-ac-01-L0-100"],
            "L1": ["250604-ac-02-L1", "250605-ac-02-L1-100"],
            "L2": ["250604-ac-03-L2", "250605-ac-03-L2-100"],
            "real": ["250604-ac-04-real", "250605-ac-04-real-100"],
        },
        "plot_args": [
            {"color": "blue", "marker": "o", "label": r"$\textrm{Arch. } [28, 20]$"},
            {
                "color": "orange",
                "marker": "s",
                "label": r"$\textrm{Arch. } [100, 100]$",
            },
        ],
        "suffix": "_arch_{data}",
    },
}


def get_fits_and_args(mode, data_type="L0"):
    """Get the appropriate fits list and plot arguments based on mode."""
    config = COMPARISON_CONFIGS[mode]

    if mode == "data_generation":
        return config["fits"], config["plot_args"], config["suffix"]
    elif mode == "architecture":
        if data_type not in config["fits"]:
            raise ValueError(
                f"Data type '{data_type}' not available. Choose from: {list(config['fits'].keys())}"
            )
        return (
            config["fits"][data_type],
            config["plot_args"],
            config["suffix"].format(data=data_type),
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


def produce_eigvals_comparison(args):
    """Produce a comparison of eigenvalues for different fits."""
    fits, plot_args, suffix = get_fits_and_args(args.mode, args.data)

    fit_dict = {}
    for fit in fits:
        eigvals_time = load_serialized_data(
            fit, "eigvals_time", fit_path=Path(args.fit_folder)
        )
        epochs = load_serialized_data(
            fit, "common_epochs", fit_path=Path(args.fit_folder)
        )

        eigvals_mean = np.array([eigvals.get_mean(axis=0) for eigvals in eigvals_time])
        eigvals_std = np.array([eigvals.get_std(axis=0) for eigvals in eigvals_time])

        fit_dict[fit] = {
            "eigvals_mean": eigvals_mean,
            "eigvals_std": eigvals_std,
            "epochs": epochs,
        }

    for i in range(args.nvals):

        fig, axs = plt.subplots(
            2,
            1,
            figsize=(10, 5),
            sharex=True,
            gridspec_kw={
                "hspace": 0.0,
                "height_ratios": [3, 1],
            },
        )
        res = []

        for idx, (_, fit_vals) in enumerate(fit_dict.items()):
            label = plot_args[idx]["label"]
            color = plot_args[idx]["color"]

            # Absolute panel
            axs[0].plot(
                fit_vals["epochs"],
                fit_vals["eigvals_mean"][:, i],
                label=label,
                color=color,
            )
            axs[0].fill_between(
                fit_vals["epochs"],
                fit_vals["eigvals_mean"][:, i] - fit_vals["eigvals_std"][:, i],
                fit_vals["eigvals_mean"][:, i] + fit_vals["eigvals_std"][:, i],
                alpha=0.5,
                color=color,
            )

            # Relative panel
            axs[1].fill_between(
                fit_vals["epochs"],
                -fit_vals["eigvals_std"][:, i] / fit_vals["eigvals_mean"][:, i],
                +fit_vals["eigvals_std"][:, i] / fit_vals["eigvals_mean"][:, i],
                alpha=0.5,
                color=color,
            )

        # Y labels
        axs[0].set_ylabel(r"$\lambda^{(" + str(i + 1) + ")}(t)$", fontsize=FONTSIZE)
        axs[1].set_ylabel(
            r"$\delta \lambda^{(" + str(i + 1) + ")}(t)$", fontsize=FONTSIZE
        )

        # Plot horizontal line at 0
        axs[1].axhline(0, color="black", linestyle="--", linewidth=0.5)

        # X label
        axs[1].set_xlabel(r"${\rm Epoch}$", fontsize=FONTSIZE)

        # Tick size
        axs[0].tick_params(labelsize=TICKSIZE)
        axs[1].tick_params(labelsize=TICKSIZE)

        # Legend
        axs[0].legend(fontsize=LEGENDSIZE)

        fig.tight_layout()

        # Generate the filename with appropriate suffix
        filename = f"eigval_{i+1}{suffix}.pdf"
        fig.savefig(PLOT_DIR / filename, bbox_inches="tight")

        res.append((fig, axs))

    return res


def produce_delta_ntk_comparison(args):
    """Produce a comparison of delta NTK for different fits."""
    fits, plot_args, suffix = get_fits_and_args(args.mode, args.data)

    fit_dict = {}
    for fit in fits:
        ntk_time = load_serialized_data(fit, "NTK_time", fit_path=Path(args.fit_folder))
        epochs = load_serialized_data(
            fit, "common_epochs", fit_path=Path(args.fit_folder)
        )

        Delta_ntk_t = []
        replicas = ntk_time[0].size
        for i in range(len(ntk_time) - 1):
            delta_ntk_dist = Distribution(f"Delta NTK {i}")
            for rep in range(replicas):
                delta_ntk = np.linalg.norm(
                    ntk_time[i + 1][rep] - ntk_time[i][rep]
                ) / np.linalg.norm(ntk_time[i][rep])
                delta_ntk_dist.add(delta_ntk)
            Delta_ntk_t.append(delta_ntk_dist)

        delta_ntk_means = np.array([delta.get_mean() for delta in Delta_ntk_t])
        delta_ntk_stds = np.array([delta.get_std() for delta in Delta_ntk_t])

        fit_dict[fit] = {
            "ntk_mean": delta_ntk_means,
            "ntk_std": delta_ntk_stds,
            "epochs": epochs,
        }

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for idx, (_, fit_vals) in enumerate(fit_dict.items()):
        label = plot_args[idx]["label"]
        color = plot_args[idx]["color"]
        ax.plot(fit_vals["epochs"][1:], fit_vals["ntk_mean"], label=label, color=color)
        ax.fill_between(
            fit_vals["epochs"][1:],
            fit_vals["ntk_mean"] - fit_vals["ntk_std"],
            fit_vals["ntk_mean"] + fit_vals["ntk_std"],
            alpha=0.5,
            color=color,
        )

    ax.set_xlabel(r"${\rm Epoch}$", fontsize=FONTSIZE)
    ax.set_ylabel(r"$\delta \Theta_t$", fontsize=FONTSIZE)
    ax.legend(fontsize=LEGENDSIZE)

    fig.tight_layout()

    # Generate the filename with appropriate suffix
    filename = f"delta_ntk{suffix}.pdf"
    fig.savefig(PLOT_DIR / filename, bbox_inches="tight")

    return fig, ax


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
        "--nvals", "-n", type=int, default=5, help="Number of eigenvalues to plot"
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["data_generation", "architecture"],
        default="data_generation",
        help="Comparison mode: 'data_generation' for L0/L1/L2, 'architecture' for different architectures",
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default="L0",
        help="Data generation (L0, L1, L2, or real) - only used in 'architecture' mode",
    )
    args = parser.parse_args()

    # Validate arguments
    if (
        args.mode == "architecture"
        and args.data not in COMPARISON_CONFIGS["architecture"]["fits"]
    ):
        available_data = list(COMPARISON_CONFIGS["architecture"]["fits"].keys())
        parser.error(f"For architecture mode, --data must be one of: {available_data}")

    _ = produce_eigvals_comparison(args)
    _ = produce_delta_ntk_comparison(args)


if __name__ == "__main__":
    main()
