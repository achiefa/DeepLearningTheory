from argparse import ArgumentParser

import numpy as np
import yaml

from yadlt.distribution import Distribution, combine_distributions
from yadlt.evolution import EvolutionOperatorComputer
from yadlt.plotting import produce_plot


def main():
    parser = ArgumentParser(description="Compute and plot PDFs.")
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

    distribution_grids = []
    epochs = None

    for idx, fitname in enumerate(fitnames):
        evolution = EvolutionOperatorComputer(fitname)
        data_type = evolution.metadata["arguments"]["data"]
        replicas = evolution.replicas

        ntk_by_time = evolution.NTK_time

        delta_ntk_t = []
        for i in range(len(ntk_by_time) - 1):
            delta_ntk_dist = Distribution(f"Delta NTK {i}")
            for rep in range(replicas):
                delta_ntk = np.linalg.norm(
                    ntk_by_time[i + 1][rep] - ntk_by_time[i][rep]
                ) / np.linalg.norm(ntk_by_time[i][rep])
                delta_ntk_dist.add(delta_ntk)
            delta_ntk_t.append(delta_ntk_dist)

        if epochs is None:
            epochs = evolution.epochs
        else:
            assert np.array_equal(
                epochs, evolution.epochs
            ), "Epochs do not match across fits."

        delta_ntk_distribution_by_epoch = combine_distributions(delta_ntk_t)
        if fitlabels is None:
            delta_ntk_distribution_by_epoch.set_name(rf"$\textrm{{{data_type}}}$")
        else:
            delta_ntk_distribution_by_epoch.set_name(rf"{fitlabels[idx]}")

        distribution_grids.append(delta_ntk_distribution_by_epoch)

    produce_plot(
        epochs[1::],
        distribution_grids,
        scale="linear",
        xlabel=r"${\rm Epoch}$",
        ylabel=r"$\delta \Theta$",
        filename=args.filename,
        save_fig=True,
        labels=fitlabels,
        colors=fitcolors,
    )


if __name__ == "__main__":
    main()
