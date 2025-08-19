"""Module to plot the kernel from the recursion relation."""

from typing import List

import keras
import numpy as np

from yadlt.model import compute_K_by_layer, compute_kernel_from_recursion
from yadlt.plotting.plotting import produce_mat_plot


def plot_kernel_at_layer(
    model_ens: tuple[keras.Model],
    layer_idx: int,
    input_data: np.ndarray,
    num_samples: int = 1000,
    batch_size: int = 100,
    diag_idx: int = 0,
    offdiag_idx: int = None,
    plot_kwargs_list: List[dict] = None,
) -> None:
    """Plot the kernel at a specific layer."""

    K_empirical = compute_K_by_layer(model_ens, layer_idx, input_data)
    K_theory = compute_kernel_from_recursion(
        tuple(model_ens),
        layer_idx,
        tuple(input_data),
        num_samples=num_samples,
        batch_size=batch_size,
    )

    if diag_idx < K_empirical.shape[1]:
        K_diag = K_empirical[:, diag_idx, :, diag_idx]
        produce_mat_plot(
            [K_diag, K_theory],
            [
                rf"$K^{{({layer_idx})}}~\rm{{empirical}}$",
                rf"$K^{{({layer_idx})}}~\rm{{theory}}$",
            ],
            **plot_kwargs_list[0],
        )
    else:
        raise ValueError(
            f"diag_idx {diag_idx} is out of bounds for K_empirical with shape {K_empirical.shape}."
        )

    if offdiag_idx is not None and offdiag_idx < K_empirical.shape[1]:
        K_offdiag = K_empirical[:, diag_idx, :, offdiag_idx]
        produce_mat_plot(
            [K_offdiag, K_theory],
            [
                rf"$K^{{({layer_idx})}}~\rm{{empirical}}$",
                rf"$K^{{({layer_idx})}}~\rm{{theory}}$",
            ],
            **plot_kwargs_list[1],
        )
