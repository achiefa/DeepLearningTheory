from typing import List

import numpy as np

from yadlt.context import FitContext
from yadlt.distribution import Distribution, combine_distributions
from yadlt.plotting.plotting import produce_plot


def plot_delta_ntk(
    fitnames: List[str], fitlabels: List[str], fitcolors: List[str], **plot_kwargs
) -> None:
    """Plot the evolution of the delta NTK."""
    distribution_grids = []
    epochs = None

    for idx, fitname in enumerate(fitnames):
        context = FitContext(fitname)

        data_type = context.get_config("metadata", "arguments")["data"]
        replicas = context.get_property("nreplicas")

        ntk_by_time = context.NTK_time

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
            epochs = context.get_config("replicas", "common_epochs")
        else:
            assert np.array_equal(
                epochs, context.get_config("replicas", "common_epochs")
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
        labels=fitlabels,
        colors=fitcolors,
        **plot_kwargs,
    )
