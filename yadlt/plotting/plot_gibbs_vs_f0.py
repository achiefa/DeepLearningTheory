import numpy as np

from yadlt.context import FitContext
from yadlt.distribution import Distribution
from yadlt.plotting.plotting import produce_mat_plot
from yadlt.utils import gibbs_fn, produce_model_at_initialisation

SIGMA = 0.25
L0 = 1.7
DELTA = 1.0e-9
ALPHA = -0.1


def plot_gibbs_vs_f0(context: FitContext, seed: int = 0, **plot_kwargs):
    """Plot the Gibbs kernel against the covariance of f0."""
    f_init = produce_model_at_initialisation(
        context.get_property("nreplicas"),
        tuple(context.load_fk_grid()),
        tuple(context.get_config("metadata", "model_info")["architecture"]),
        seed=seed,
    )
    f0_mean = f_init.get_mean()

    # Compute covariance of f0
    cov_f0 = Distribution(
        name=r"$\textrm{Cov}[f_0, f_0]$",
        size=f0_mean.size,
        shape=(f0_mean.shape[0], f0_mean.shape[0]),
    )
    for f0_rep in f_init:
        cov = np.outer(f0_rep, f0_rep)
        cov_f0.add(cov)

    # Produce comparison with Gibbs kernel
    fk_grid = context.load_fk_grid()
    gibbs_kernel = np.fromfunction(
        lambda i, j: gibbs_fn(fk_grid[i], fk_grid[j], DELTA, SIGMA, L0),
        (fk_grid.size, fk_grid.size),
        dtype=int,
    )

    produce_mat_plot(
        matrices=[gibbs_kernel, cov_f0.get_mean()],
        titles=[r"$\textrm{Gibbs}$", r"$\textrm{Cov}[f_0, f_0]$"],
        **plot_kwargs
    )
