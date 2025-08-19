"""Module for plotting eigenvalues of NTK or H in a single plot."""

from typing import List

from yadlt.context import FitContext
from yadlt.distribution import combine_distributions
from yadlt.plotting.plotting import produce_plot


def plot_eigvals_single_plot(
    fitnames: List[str],
    labels: List[str],
    colors: List[str],
    eigvals: List[int] = [0],
    eigval_type: str = "ntk",
    ylabel: str = r"$\textrm{Eigenvalues}$",
    group_by: str = "eigvals",
    **plot_kwargs,
) -> None:
    """Plot the eigenvalues of the NTK or H in a single plot. The lines can be grouped by fitnames."""
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

        if eigval_type == "ntk":
            tmp = combine_distributions(context.eigvals_time)
        elif eigval_type == "h":
            tmp = combine_distributions(context.h_by_epoch)
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
        number_of_groups = len(fitnames)
    elif group_by == "eigvals":
        # Group by eigenvalue first (fit1_eig1, fit2_eig1, fit1_eig2, fit2_eig2, ...)
        all_slices.sort(key=lambda x: (x["eig_idx"], x["fit_idx"]))
        number_of_groups = len(eigvals)
    else:
        raise ValueError(f"Unknown group_by: {group_by}. Use 'fitnames' or 'eigvals'.")

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
        colors=colors,
        labels=labels,
        xlabel=r"${\rm Epoch}$",
        ylabel=ylabel,
        ax_specs=ax_specs,
        group_size=int(group_size),
        **plot_kwargs,
    )
