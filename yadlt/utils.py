import functools

import numpy as np

from yadlt.context import FitContext
from yadlt.distribution import Distribution
from yadlt.evolution import EvolutionOperatorComputer
from yadlt.model import generate_pdf_model, load_trained_model


def compute_distance(plotting_grids, normalize_to: int | None = 1):
    """
    Plot the distance between two PDFs normalised to the standard deviation.
    """
    normalize_to -= 1  # Convert to 1-based index
    gr2_stats = plotting_grids[normalize_to]
    cv2 = gr2_stats.get_mean()
    sg2 = gr2_stats.get_std()
    N2 = gr2_stats.size

    distances = []
    for idx, grid in enumerate(plotting_grids):
        if idx == normalize_to:
            continue

        cv1 = grid.get_mean()
        sg1 = grid.get_std()
        N1 = grid.size

        # Wrap the distance into a Stats (1, flavours, points)
        distances.append(
            (grid.name, np.sqrt((cv1 - cv2) ** 2 / (sg1**2 / N1 + sg2**2 / N2)))
        )

    return distances


def gibbs_fn(x1, x2, delta, sigma, l0):
    """
    Gibbs kernel function for two points x1 and x2 with parameters delta, sigma, and l0.
    """

    def l(x):
        return l0 * (x + delta)

    return (
        x1
        * x2
        * sigma**2
        * np.sqrt(2 * l(x1) * l(x2) / (np.power(l(x1), 2) + np.power(l(x2), 2)))
        * np.exp(-np.power(x1 - x2, 2) / (np.power(l(x1), 2) + np.power(l(x2), 2)))
    )


@functools.cache
def load_and_evaluate_model(context, epoch=-1):
    # Extract the last common epoch from the ensemble of replicas
    common_epochs = context.get_config("replicas", "common_epochs")
    replicas_folders = context.get_config("folders", "replicas_folders")
    epoch_idx = -1 if epoch == -1 else common_epochs.index(epoch)

    xT3_training = Distribution("xT3_training")

    grid = context.load_fk_grid()
    x = np.array(grid).reshape(1, -1, 1)

    for replica_path in replicas_folders:
        epoch = common_epochs[epoch_idx]
        model, _ = load_trained_model(replica_path, epoch)
        xT3_training.add(model(x).numpy().reshape(-1))

    return xT3_training


@functools.cache
def load_data(context: FitContext):
    # Load the data used to fit the replicas
    data_by_replica_original = Distribution("Original replicas of the data")

    for rep in range(context.get_property("nreplicas")):
        data = context.get_data_by_replica(rep)
        data_by_replica_original.add(data)
    return data_by_replica_original


@functools.cache
def produce_model_at_initialisation(
    replicas, fk_grid_tuple, architecture_tuple, seed: int = 0
):
    """Initialise a model at random initialisation."""
    xT3_0 = Distribution("xT3 at initialisation")
    x = np.array(fk_grid_tuple).reshape(1, -1, 1)
    for rep in range(replicas):
        model = generate_pdf_model(
            outputs=1,
            architecture=list(architecture_tuple),
            activations=["tanh", "tanh"],
            kernel_initializer="GlorotNormal",
            user_ki_args=None,
            seed=seed + rep,
            scaled_input=False,
            preprocessing=False,
        )

        xT3_0.add(model(x).numpy().reshape(-1))

    return xT3_0


def evaluate_from_ref_function(context: FitContext, ref_epoch: int = 0):
    """Utility function to compute the xT3 from a reference epoch
    of the training process using the boundary condition of the
    reference epoch."""
    # Load trained solution (at reference epoch)
    xT3_ref = load_and_evaluate_model(context, ref_epoch)
    xT3_ref.set_name(r"$\textrm{TS @ reference epoch}$")

    data_by_replica_original = load_data(context)

    evolution = EvolutionOperatorComputer(context)

    def func(t: float):
        U, V = evolution.compute_evolution_operator(ref_epoch, t)
        xT3_t = U @ xT3_ref + V @ data_by_replica_original
        return xT3_t

    return func


def evaluate_from_initialisation(
    context: FitContext, ref_epoch: int = 0, seed: int = 0
):
    """Utility function to compute the xT3 from a random initialisation
    using a frozen NTK specified by the reference epoch."""
    replicas = context.get_property("nreplicas")
    fk_grid = context.load_fk_grid()
    arch_tuple = tuple(context.get_config("metadata", "model_info")["architecture"])

    xT3_0 = produce_model_at_initialisation(
        replicas=replicas,
        fk_grid_tuple=tuple(fk_grid),
        architecture_tuple=arch_tuple,
        seed=seed,
    )
    data_by_replica_original = load_data(context)

    evolution = EvolutionOperatorComputer(context)

    def func(t: float):
        # Load data
        U, V = evolution.compute_evolution_operator(ref_epoch, t)
        xT3_t = U @ xT3_0 + V @ data_by_replica_original
        return xT3_t

    return func


def compute_covariance_decomposition_by_t(
    context: FitContext, ref_epoch: int, f_0: Distribution, divide_by_x: bool = False
):
    """Compute the covariance decomposition of the analytical solution according to
    the expression reported in the paper:

      Cov[f_t, f_t] = C_t^{(00)} + C_t^{(0Y)} + C_t^{(YY)}.

    Args:
        context (FitContext): The fit context containing relevant information.
        ref_epoch (int): The reference epoch to compute the covariance at.
        f_0 (Distribution): The distribution at the initialisation.
    Returns:
        func (callable): A function that computes the covariance decomposition at a given time t.
        The function returns C_00, C_0Y, and C_YY, in that order.
    """
    evolution = EvolutionOperatorComputer(context)
    Y = load_data(context)

    def func(t: float):
        # Compute evolution operators
        U, V = evolution.compute_evolution_operator(ref_epoch, t)

        # Compute objects for the covariance decomposition
        U_f0 = U @ f_0
        V_Y = V @ Y

        if divide_by_x:
            fk_grid = context.load_fk_grid()
            U_f0 = U_f0 / fk_grid
            V_Y = V_Y / fk_grid

        C_00 = (U_f0 ^ U_f0).get_mean() - np.outer(U_f0.get_mean(), U_f0.get_mean())
        C_0Y = (
            +(U_f0 ^ V_Y).get_mean()
            - np.outer(U_f0.get_mean(), V_Y.get_mean())
            + (V_Y ^ U_f0).get_mean()
            - np.outer(V_Y.get_mean(), U_f0.get_mean())
        )
        C_YY = (V_Y ^ V_Y).get_mean() - np.outer(V_Y.get_mean(), V_Y.get_mean())

        return C_00, C_0Y, C_YY

    return func


def compute_covariance_ft(
    context: FitContext, ref_epoch: int, f_0: Distribution, divide_by_x: bool = False
):
    """Compute the covariance of the analytical solution

      Cov[f_t, f_t].

    Args:
        context (FitContext): The fit context containing relevant information.
        ref_epoch (int): The reference epoch to compute the covariance at.
        f_0 (Distribution): The distribution at the initialisation.
    Returns:
        func (callable): A function that computes the covariance at a given time t.
        The function returns Cov[f_t, f_t].
    """
    evolution = EvolutionOperatorComputer(context)
    Y = load_data(context)

    def func(t: float):
        # Compute evolution operators
        U, V = evolution.compute_evolution_operator(ref_epoch, t)

        U_f0 = U @ f_0
        V_Y = V @ Y

        res = U_f0 + V_Y

        if divide_by_x:
            fk_grid = context.load_fk_grid()
            res = res / fk_grid

        C = (res ^ res).get_mean() - np.outer(res.get_mean(), res.get_mean())
        return C

    return func


def compute_covariance_training(
    context: FitContext, epoch: int = 0, divide_by_x: bool = False
):
    """Compute the covariance of the trained solution at a given epoch

      Cov[f_t, f_t].

    Args:
        context (FitContext): The fit context containing relevant information.
        epoch (int): The epoch to compute the covariance at.
    Returns:
        mat (ndarray): The covariance at a given time t.
    """
    ft = load_and_evaluate_model(context, epoch)
    if divide_by_x:
        fk_grid = context.load_fk_grid()
        ft = ft / fk_grid
    C = (ft ^ ft).get_mean() - np.outer(ft.get_mean(), ft.get_mean())
    return C
