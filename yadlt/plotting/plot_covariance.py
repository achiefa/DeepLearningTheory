"""Module to plot the covariance matrix."""

from typing import List

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

from yadlt import evolution
from yadlt.context import FitContext
from yadlt.plotting.plotting import next_color, produce_errorbar_plot, produce_mat_plot
from yadlt.utils import (
    compute_covariance_decomposition_by_t,
    compute_covariance_ft,
    compute_covariance_training,
    evaluate_from_initialisation,
    load_and_evaluate_model,
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
    divide_by_x: bool = False,
    normalize: bool = False,
    **plot_kwargs,
):
    """Plot the diagonal uncertainty of the decomposition of the covariance matrix for the analytical solution.

    Args:
        context (FitContext): The fit context containing relevant information.
        ref_epoch (int): The reference epoch to compute the covariance at.
        epoch (int): The epoch to compute the covariance at.
        seed (int): The seed to use for the initialisation.
        common_plt_spec (dict): A dictionary containing common plot specifications.
        divide_by_x (bool): Whether to divide the covariance by x.
        normalize (bool): Whether to normalize the covariance by the mean of the prediction.
        plot_kwargs (dict): Additional keyword arguments to pass to the plotting function.
    """
    learning_rate = float(context.get_config("metadata", "arguments")["learning_rate"])

    f0 = produce_model_at_initialisation(
        context.get_property("nreplicas"),
        tuple(context.load_fk_grid()),
        tuple(context.get_config("metadata", "model_info")["architecture"]),
        seed=seed,
    )

    # Compute covariance matrices
    cov_decomposition = compute_covariance_decomposition_by_t(
        context, ref_epoch=ref_epoch, f_0=f0, divide_by_x=divide_by_x
    )
    cov_ft = compute_covariance_ft(
        context, ref_epoch=ref_epoch, f_0=f0, divide_by_x=divide_by_x
    )

    x = (
        context.load_fk_grid()
    )  # [i + 1 for i in range(context.load_fk_grid().shape[0])]
    box_x_size = [0.2 * (fk_p - fk_m) for fk_m, fk_p in zip(x[0:], x[1:])]
    box_x_size.append(box_x_size[-1])

    evolution_time = epoch * learning_rate
    C_00, _, C_YY = cov_decomposition(evolution_time)
    C = cov_ft(evolution_time)
    zeros = np.zeros_like(C.diagonal())

    if normalize:
        f = evaluate_from_initialisation(context, ref_epoch=ref_epoch, f0=f0)
        ft = f(evolution_time)
        # full_cov_diag = np.sqrt(C.diagonal()) / np.abs(ft.get_mean())
        c00 = np.sqrt(C_00.diagonal()) / np.abs(ft.get_mean())
        cyy = np.sqrt(C_YY.diagonal()) / np.abs(ft.get_mean())
    else:
        # full_cov_diag = np.sqrt(C.diagonal())
        c00 = np.sqrt(C_00.diagonal())
        cyy = np.sqrt(C_YY.diagonal())

    # Loop over data points; create box from errors at each point
    # errorboxes = [
    #     Rectangle((xp - x_size, yp - ye), x_size * 2, ye * 2)
    #     for xp, yp, ye, x_size in zip(x, zeros, full_cov_diag, box_x_size)
    # ]

    # # Create patch collection with specified colour/alpha
    # pc = PatchCollection(errorboxes, facecolor="grey", alpha=0.2, edgecolor="none")

    # Add collection to Axes
    produce_errorbar_plot(
        xgrid=x,
        add_grids=[
            {
                "label": r"$C_t^{(00)}$",
                "mean": zeros,
                "std": c00,
                "spec": common_plt_spec | {"color": "C1"},
            },
            {
                "label": r"$C_t^{(YY)}$",
                "mean": zeros,
                "std": cyy,
                "spec": common_plt_spec | {"color": "C2"},
            },
            # {"label": r"$C_t^{(0Y)}$", "mean": zeros, "std": np.sqrt(C_0Y.diagonal()), "spec": common_plt_spec | {"color": "C3"}},
            # {"label": r"$\rm{Cov}[f_t, f_t]$", "box": pc},
        ],
        **plot_kwargs,
    )


def plot_diag_error_compare_an_tr(
    context: FitContext,
    ref_epoch: int = 0,
    epoch: int = 0,
    seed: int = 0,
    common_plt_spec: dict = {},
    divide_by_x: bool = False,
    normalize: bool = False,
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

    # Compute evolution time
    evolution_time = epoch * learning_rate

    # Compute covariance matrices
    cov_ft = compute_covariance_ft(
        context, ref_epoch=ref_epoch, f_0=f0, divide_by_x=divide_by_x
    )
    cov_f_trained = compute_covariance_training(
        context, epoch=epoch, divide_by_x=divide_by_x
    )
    C = cov_ft(evolution_time)

    x = (
        context.load_fk_grid()
    )  # [i + 1 for i in range(context.load_fk_grid().shape[0])]
    box_x_size = [0.2 * (fk_p - fk_m) for fk_m, fk_p in zip(x[0:], x[1:])]
    box_x_size.append(box_x_size[-1])

    zeros = np.zeros_like(C.diagonal())

    if normalize:
        f = evaluate_from_initialisation(context, ref_epoch=ref_epoch, f0=f0)
        ft = f(evolution_time)
        ft_trained = load_and_evaluate_model(context, epoch)
        an_cov_diag = np.sqrt(C.diagonal()) / np.abs(ft.get_mean())
        tr_cov_diag = np.sqrt(cov_f_trained.diagonal()) / np.abs(ft_trained.get_mean())
    else:
        an_cov_diag = np.sqrt(C.diagonal())
        tr_cov_diag = np.sqrt(cov_f_trained.diagonal())

    # Add collection to Axes
    produce_errorbar_plot(
        xgrid=x,
        add_grids=[
            {
                "label": r"$\rm{Cov}[f_t^{\rm (an)}, f_t^{\rm (an)}]$",
                "mean": zeros,
                "std": an_cov_diag,
                "spec": common_plt_spec | {"color": "C1"},
            },
            {
                "label": r"$\rm{Cov}[f_t^{\rm (tr)}, f_t^{\rm (tr)}]$",
                "mean": zeros,
                "std": tr_cov_diag,
                "spec": common_plt_spec | {"color": "C2"},
            },
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
