"""Module to plot the evolution of eigenvalues of the NTK."""

from typing import List

from yadlt.context import FitContext
from yadlt.distribution import combine_distributions
from yadlt.plotting.plotting import produce_pdf_plot


def plot_eigvals(
    fitnames: List[str],
    fitlabels: List[str],
    fitcolors: List[str],
    eigvals: List[int],
    filename_prefix: str = "ntk_eigvals",
    **plot_kwargs,
) -> None:
    """Plot the evolution of the single eigenvalues of the NTK."""
    eigvals_by_fit = []
    epochs = None

    for fitname in fitnames:
        context = FitContext(fitname)
        if epochs is None:
            epochs = context.get_config("replicas", "common_epochs")
        else:
            assert epochs == context.get_config(
                "replicas", "common_epochs"
            ), "Epochs do not match across fits."
        tmp = combine_distributions(context.eigvals_time)
        tmp.set_name(fitname)
        eigvals_by_fit.append(tmp)

    for idx_eig in eigvals:
        eigvals_grids = []
        for eigval in eigvals_by_fit:
            sliced_distribution = eigval.slice((slice(None), idx_eig - 1))
            eigvals_grids.append(sliced_distribution)

        filename = f"{filename_prefix}_n_{idx_eig}.pdf"

        produce_pdf_plot(
            epochs,
            eigvals_grids,
            normalize_to=0,
            xlabel=r"${\rm Epoch}$",
            ylabel=r"$\lambda^{(" + str(idx_eig) + ")}$",
            ratio_label=r"$\delta \lambda^{(" + str(idx_eig) + ")}$",
            labels=fitlabels,
            colors=fitcolors,
            filename=filename,
            **plot_kwargs,
        )
