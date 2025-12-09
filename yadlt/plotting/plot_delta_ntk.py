from typing import List

import numpy as np

from yadlt.context import FitContext
from yadlt.plotting.plotting import produce_plot
from yadlt.utils import compute_delta_ntk


def plot_delta_ntk(
    fitnames: List[str], fitlabels: List[str], fitcolors: List[str], **plot_kwargs
) -> None:
    """Plot the evolution of the delta NTK."""
    distribution_grids = []
    epochs = None

    for idx, fitname in enumerate(fitnames):
        context = FitContext(fitname)

        data_type = context.get_config("metadata", "arguments")["data"]

        delta_ntk_distribution_by_epoch = compute_delta_ntk(context)

        if epochs is None:
            epochs = context.get_config("replicas", "common_epochs")
        else:
            assert np.array_equal(
                epochs, context.get_config("replicas", "common_epochs")
            ), "Epochs do not match across fits."

        if fitlabels is None:
            delta_ntk_distribution_by_epoch.set_name(rf"$\textrm{{{data_type}}}$")
        else:
            delta_ntk_distribution_by_epoch.set_name(rf"{fitlabels[idx]}")

        distribution_grids.append(delta_ntk_distribution_by_epoch)

    produce_plot(
        epochs[1::],
        distribution_grids,
        xlabel=r"${\rm Epoch}$",
        ylabel=r"$\delta \Theta$",
        labels=fitlabels,
        colors=fitcolors,
        **plot_kwargs,
    )
