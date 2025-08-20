"""Module to plot contributions of U and V in the evolution operator."""

from yadlt.context import FitContext
from yadlt.evolution import EvolutionOperatorComputer
from yadlt.plotting.plotting import produce_pdf_plot
from yadlt.utils import (
    load_and_evaluate_model,
    load_data,
    produce_model_at_initialisation,
)


def plot_u_v_contributions(
    context: FitContext,
    ref_epoch: int = 0,
    ev_epoch: int = 1000,
    seed: int = 0,
    **plot_kwargs,
):
    """Plot the contributions of U and V to the total operator."""
    nreplicas = context.get_property("nreplicas")
    architecture = tuple(context.get_config("metadata", "model_info")["architecture"])
    common_epochs = context.get_config("replicas", "common_epochs")
    fk_grid = context.load_fk_grid()

    # Load trained solution (end of training)
    xT3_training = load_and_evaluate_model(context, -1)
    xT3_training.set_name(rf"$\textrm{{TS @ {common_epochs[-1]}}}$")

    # Load data
    data_by_replica_original = load_data(context)

    # Get learning rate
    learning_rate = float(context.get_config("metadata", "arguments")["learning_rate"])
    t = ev_epoch * learning_rate

    # Load trained solution (at reference epoch)
    # xT3_ref = load_model(context, ref_epoch)
    # xT3_ref.set_name(rf"$\textrm{{TS @ }} T_{{\rm ref}}$")
    xT3_0 = produce_model_at_initialisation(
        replicas=nreplicas,
        fk_grid_tuple=tuple(fk_grid),
        architecture_tuple=architecture,
        seed=seed,
    )

    evolution = EvolutionOperatorComputer(context)

    # Evolve the solution
    U, V = evolution.compute_evolution_operator(ref_epoch, t)
    xT3_t_u = U @ xT3_0
    xT3_t_u.set_name(r"$\textrm{Contribution from U}$")
    xT3_t_v = V @ data_by_replica_original
    xT3_t_v.set_name(r"$\textrm{Contribution from V}$")

    ax_specs_ratio = {"set_ylim": (0.8, 1.2)}

    produce_pdf_plot(
        fk_grid,
        [xT3_training, xT3_t_u, xT3_t_v],
        ax_specs=[None, ax_specs_ratio],
        normalize_to=1,
        xlabel=r"$x$",
        ylabel=r"$xT_3(x)$",
        title=rf"$T_{{\rm ref}} = {{{ref_epoch}}}, \quad f_0 = f^{{(\rm init)}}, \quad T = {{{ev_epoch}}}$",
        **plot_kwargs,
    )
