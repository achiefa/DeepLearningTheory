"""Module to plot the covariance matrix."""

from typing import List

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import numpy as np

from yadlt.context import FitContext
from yadlt.plotting.plotting import next_color, produce_errorbar_plot, produce_mat_plot
from yadlt.utils import (
    compute_covariance_decomposition_by_t,
    compute_covariance_ft,
    compute_covariance_training,
    produce_model_at_initialisation,
)


def plot_covariance_decomposition_by_epoch(
    context: FitContext,
    ref_epoch: int = 0,
    epoch: int = 0,
    seed: int = 0,
    **plot_kwargs,
):
    """Plot the decomposition of the covariance matrix for the analytical solution"""
    learning_rate = float(context.get_config("metadata", "arguments")["learning_rate"])

    f0 = produce_model_at_initialisation(
        context.get_property("nreplicas"),
        tuple(context.load_fk_grid()),
        tuple(context.get_config("metadata", "model_info")["architecture"]),
        seed=seed,
    )

    # Compute covariance matrices
    cov_decomposition = compute_covariance_decomposition_by_t(
        context, ref_epoch=ref_epoch, f_0=f0
    )
    cov_ft = compute_covariance_ft(context, ref_epoch=ref_epoch, f_0=f0)

    evolution_time = epoch * learning_rate
    C_00, C_0Y, C_YY = cov_decomposition(evolution_time)
    C = cov_ft(evolution_time)

    produce_mat_plot(
        [C, C_00, C_0Y, C_YY],
        [r"$\rm{Cov}[f_t, f_t]$", r"$C_t^{(00)}$", r"$C_t^{(0Y)}$", r"$C_t^{(YY)}$"],
        **plot_kwargs,
    )


def plot_covariance_decomposition(
    context: FitContext,
    ref_epoch: int = 0,
    epochs: List[int] = 0,
    seed: int = 0,
    **plot_kwargs,
):
    """Plot the decomposition of the covariance matrix for the analytical solution"""
    learning_rate = float(context.get_config("metadata", "arguments")["learning_rate"])
    datatype = context.get_config("metadata", "arguments")["data"]

    f0 = produce_model_at_initialisation(
        context.get_property("nreplicas"),
        tuple(context.load_fk_grid()),
        tuple(context.get_config("metadata", "model_info")["architecture"]),
        seed=seed,
    )

    # Compute covariance matrices
    cov_decomposition = compute_covariance_decomposition_by_t(
        context, ref_epoch=ref_epoch, f_0=f0
    )
    cov_ft = compute_covariance_ft(context, ref_epoch=ref_epoch, f_0=f0)

    list_of_matrices = []
    for epoch in epochs:
        evolution_time = epoch * learning_rate
        C_00, C_0Y, C_YY = cov_decomposition(evolution_time)
        C = cov_ft(evolution_time)
        list_of_matrices.append((C, C_00, C_0Y, C_YY))

    vmin = min([np.percentile(A, 1) for ls in list_of_matrices for A in ls])
    vmax = max([np.percentile(A, 95) for ls in list_of_matrices for A in ls])

    for epoch, (C, C_00, C_0Y, C_YY) in zip(epochs, list_of_matrices):
        produce_mat_plot(
            [C, C_00, C_0Y, C_YY],
            [
                r"$\rm{Cov}[f_t, f_t]$",
                r"$C_t^{(00)}$",
                r"$C_t^{(0Y)}$",
                r"$C_t^{(YY)}$",
            ],
            vmin=vmin,
            vmax=vmax,
            filename=f"covariance_ft_{epoch}_{datatype}.pdf",
            text_dict={"x": -0.5, "y": 0.5, "s": rf"$T = {epoch}$"},
            **plot_kwargs,
        )


def plot_diag_error_decomposition(
    context: FitContext,
    ref_epoch: int = 0,
    epoch: int = 0,
    seed: int = 0,
    common_plt_spec: dict = {},
    **plot_kwargs,
):
    """Plot the diagonal uncertainty of the decomposition of the covariance matrix for the analytical solution"""
    learning_rate = float(context.get_config("metadata", "arguments")["learning_rate"])

    f0 = produce_model_at_initialisation(
        context.get_property("nreplicas"),
        tuple(context.load_fk_grid()),
        tuple(context.get_config("metadata", "model_info")["architecture"]),
        seed=seed,
    )

    # Compute covariance matrices
    cov_decomposition = compute_covariance_decomposition_by_t(
        context, ref_epoch=ref_epoch, f_0=f0
    )
    cov_ft = compute_covariance_ft(context, ref_epoch=ref_epoch, f_0=f0)

    x = [i + 1 for i in range(context.load_fk_grid().shape[0])]
    box_x_size = 0.5

    evolution_time = epoch * learning_rate
    C_00, C_0Y, C_YY = cov_decomposition(evolution_time)
    C = cov_ft(evolution_time)
    zeros = np.zeros_like(C.diagonal())

    full_cov_diag = np.sqrt(C.diagonal())

    # Loop over data points; create box from errors at each point
    errorboxes = [
        Rectangle((xp - box_x_size, yp - ye), box_x_size * 2, ye * 2)
        for xp, yp, ye in zip(x, zeros, full_cov_diag)
    ]

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor="grey", alpha=0.2, edgecolor="none")

    # Add collection to Axes
    produce_errorbar_plot(
        xgrid=x,
        add_grids=[
            {
                "label": r"$C_t^{(00)}$",
                "mean": zeros,
                "std": np.sqrt(C_00.diagonal()),
                "spec": common_plt_spec | {"color": "C1"},
            },
            {
                "label": r"$C_t^{(YY)}$",
                "mean": zeros,
                "std": np.sqrt(C_YY.diagonal()),
                "spec": common_plt_spec | {"color": "C2"},
            },
            # {"label": r"$C_t^{(0Y)}$", "mean": zeros, "std": np.sqrt(C_0Y.diagonal()), "spec": common_plt_spec | {"color": "C3"}},
            {"label": r"$\rm{Cov}[f_t, f_t]$", "box": pc},
        ],
        **plot_kwargs,
    )


def plot_cov_compare_tr_an(
    context: FitContext,
    ref_epoch: int = 0,
    epochs: List[int] = 0,
    seed: int = 0,
    **plot_kwargs,
):
    """Plot the decomposition of the covariance matrix for the analytical solution"""
    learning_rate = float(context.get_config("metadata", "arguments")["learning_rate"])
    datatype = context.get_config("metadata", "arguments")["data"]

    f0 = produce_model_at_initialisation(
        context.get_property("nreplicas"),
        tuple(context.load_fk_grid()),
        tuple(context.get_config("metadata", "model_info")["architecture"]),
        seed=seed,
    )

    # Compute covariance matrices
    cov_ft = compute_covariance_ft(context, ref_epoch=ref_epoch, f_0=f0)

    list_of_matrices = []
    for epoch in epochs:
        evolution_time = epoch * learning_rate
        cov_f_trained = compute_covariance_training(context, epoch=epoch)
        C = cov_ft(evolution_time)
        list_of_matrices.append((C, cov_f_trained))

    vmin = min([np.percentile(A, 1) for ls in list_of_matrices for A in ls])
    vmax = max([np.percentile(A, 95) for ls in list_of_matrices for A in ls])

    for epoch, (C, cov_f_trained) in zip(epochs, list_of_matrices):
        produce_mat_plot(
            [C, cov_f_trained],
            [r"$\rm{Cov}[f_t, f_t]$", r"$\rm{Cov}[f_t^{\rm (tr)}, f_t^{\rm (tr)}]$"],
            vmin=vmin,
            vmax=vmax,
            text_dict={"x": -0.5, "y": 0.5, "s": rf"$T = {epoch}$"},
            filename=f"cov_comparison_an_tr_{epoch}_{datatype}.pdf",
            **plot_kwargs,
        )
