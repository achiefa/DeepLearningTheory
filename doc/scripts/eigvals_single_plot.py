from argparse import ArgumentParser

import yaml

from yadlt.distribution import Distribution, combine_distributions
from yadlt.evolution import EvolutionOperatorComputer
from yadlt.plotting import produce_plot


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
    fitlabels = config.get("fitlabels", None)
    fitcolors = config.get("fitcolors", None)
    eigvals = config.get("eigvals", [0])
    eigval_type = config.get("eigval_type", None)
    ylabel = config.get("ylabel", r"")

    group_by = config["group_by"]

    eigvals_by_fit = []
    epochs = None

    for fitname in fitnames:
        evolution = EvolutionOperatorComputer(fitname)
        if epochs is None:
            epochs = evolution.epochs
        else:
            assert epochs == evolution.epochs, "Epochs do not match across fits."

        if eigval_type == "ntk":
            tmp = combine_distributions(evolution.eigvals_time)
        elif eigval_type == "h":
            tmp = combine_distributions(evolution.h_by_epoch)
        else:
            raise ValueError(f"Unknown eigval_type: {eigval_type}. Use 'ntk' or 'h'.")
        tmp.set_name(fitname)
        eigvals_by_fit.append(tmp)

    # Generate all sliced distributions with metadata
    all_slices = [
        {
            "fit_idx": fit_idx,
            "fit_name": eigval.name,
            "eig_idx": idx_eig,
            "distribution": eigval.slice((slice(None), idx_eig - 1)),
        }
        for fit_idx, eigval in enumerate(eigvals_by_fit)
        for idx_eig in eigvals
    ]

    # Reorganize based on preference
    if group_by == "fitnames":
        # Group by fit first (fit1_eig1, fit1_eig2, fit2_eig1, fit2_eig2, ...)
        all_slices.sort(key=lambda x: (x["fit_idx"], x["eig_idx"]))
    elif group_by == "eigvals":
        # Group by eigenvalue first (fit1_eig1, fit2_eig1, fit1_eig2, fit2_eig2, ...)
        all_slices.sort(key=lambda x: (x["eig_idx"], x["fit_idx"]))
    else:
        raise ValueError(f"Unknown group_by: {group_by}. Use 'fitnames' or 'eigvals'.")

    number_of_groups = len(config[group_by])
    group_size = len(all_slices) / number_of_groups
    if group_size != int(group_size):
        raise ValueError(
            f"Number of groups ({number_of_groups}) does not evenly divide the number of eigvals ({len(all_slices)})."
        )

    # Extract distributions in the desired order
    eigvals_grids = [item["distribution"] for item in all_slices]

    ax_specs = {
        "set_yscale": "log",
        "grid": {"visible": True, "which": "both", "linestyle": "--", "linewidth": 0.5},
    }

    produce_plot(
        epochs,
        eigvals_grids,
        scale="linear",
        colors=fitcolors,
        labels=fitlabels,
        xlabel=r"${\rm Epoch}$",
        ylabel=ylabel,
        ax_specs=ax_specs,
        filename=args.filename,
        save_fig=True,
        group_size=int(group_size),
    )


if __name__ == "__main__":
    main()
