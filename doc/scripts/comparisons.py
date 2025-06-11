#!/usr/bin/env python3

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


def load_serialized_data(fit_name, data_name, fit_path=FIT_PATH):
    serialization_folder = fit_path / fit_name / "serialization"
    data = pickle.load(open(serialization_folder / f"{data_name}.pickle", "rb"))
    return data


FITS = [
    "250604-ac-01-L0",
    "250604-ac-02-L1",
    "250604-ac-03-L2",
]

PLOT_ARGS = [
    {"color": "blue", "marker": "o", "label": r"$\textrm{L0}$"},
    {"color": "orange", "marker": "s", "label": r"$\textrm{L1}$"},
    {"color": "green", "marker": "D", "label": r"$\textrm{L2}$"},
]


def produce_eigvals_comparison(args):
    """Produce a comparison of eigenvalues for different fits."""
    fit_dict = {}
    for fit in FITS:
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

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        res = []

        for idx, (_, fit_vals) in enumerate(fit_dict.items()):
            label = PLOT_ARGS[idx]["label"]
            color = PLOT_ARGS[idx]["color"]
            ax.plot(
                fit_vals["epochs"],
                fit_vals["eigvals_mean"][:, i],
                label=label,
                color=color,
            )
            ax.fill_between(
                fit_vals["epochs"],
                fit_vals["eigvals_mean"][:, i] - fit_vals["eigvals_std"][:, i],
                fit_vals["eigvals_mean"][:, i] + fit_vals["eigvals_std"][:, i],
                alpha=0.5,
                color=color,
            )

        ax.set_xlabel(r"${\rm Epoch}$", fontsize=16)
        ax.set_ylabel(r"$\lambda^{(" + str(i + 1) + ")}(t)$", fontsize=16)
        ax.legend()

        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"eigval_{i+1}.pdf", bbox_inches="tight")

        res.append((fig, ax))

    return res


def produce_delta_ntk_comparison(args):
    """Produce a comparison of delta NTK for different fits."""
    fit_dict = {}
    for fit in FITS:
        ntk_time = load_serialized_data(fit, "NTK_time", fit_path=Path(args.fit_folder))
        epochs = load_serialized_data(
            fit, "common_epochs", fit_path=Path(args.fit_folder)
        )

        Delta_ntk_t = []
        replicas = ntk_time[0].size
        for i in range(len(ntk_time) - 1):
            delta_ntk_dist = Distribution(f"Delta NTK {i}")
            for rep in range(replicas):
                delta_ntk = np.linalg.norm(ntk_time[i + 1][rep] - ntk_time[i][rep])
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
        label = PLOT_ARGS[idx]["label"]
        color = PLOT_ARGS[idx]["color"]
        ax.plot(fit_vals["epochs"][1:], fit_vals["ntk_mean"], label=label, color=color)
        ax.fill_between(
            fit_vals["epochs"][1:],
            fit_vals["ntk_mean"] - fit_vals["ntk_std"],
            fit_vals["ntk_mean"] + fit_vals["ntk_std"],
            alpha=0.5,
            color=color,
        )

    ax.set_xlabel(r"${\rm Epoch}$", fontsize=16)
    ax.set_ylabel(r"$\Delta \Theta_t$", fontsize=16)
    ax.legend()

    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"delta_ntk.pdf", bbox_inches="tight")

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
    args = parser.parse_args()

    _ = produce_eigvals_comparison(args)
    _ = produce_delta_ntk_comparison(args)


if __name__ == "__main__":
    main()
