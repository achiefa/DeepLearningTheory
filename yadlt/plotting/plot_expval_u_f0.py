from yadlt.context import FitContext
from yadlt.evolution import EvolutionOperatorComputer
from yadlt.plotting.plotting import produce_plot
from yadlt.utils import produce_model_at_initialisation


def plot_expval_u_f0(
    context: FitContext,
    ref_epoch: int = 20000,
    epoch: int = 20000,
    seed: int = 0,
    **plot_kwargs,
):
    evolution = EvolutionOperatorComputer(context)
    learning_rate = float(context.get_config("metadata", "arguments")["learning_rate"])

    t = epoch * learning_rate
    U, _ = evolution.compute_evolution_operator(ref_epoch, t)

    # Produce model at initialisation
    f_init = produce_model_at_initialisation(
        context.get_property("nreplicas"),
        tuple(context.load_fk_grid()),
        tuple(context.get_config("metadata", "model_info")["architecture"]),
        seed=seed,
    )
    f_init_mean = f_init.get_mean()

    # Compute grid for U f0
    Uf0 = U @ f_init
    Uf0.set_name(r"$\mathbb{E}[U f_0]$")

    # Produce plots of the expectation value of U f0
    U_mean = U.get_mean()
    add_grid_dict = {
        "mean": U_mean @ f_init_mean,
        "spec": {
            "label": r"$\mathbb{E}[U] \mathbb{E}[f_0]$",
            "color": "black",
        },
    }

    # Produce plots of the expectation value of U f0
    produce_plot(
        context.load_fk_grid(),
        [Uf0],
        additional_grids=[add_grid_dict],
        ylabel="$U f_0$",
        xlabel="$x$",
        title=rf"$T_{{\rm ref}}={ref_epoch}, \quad f_0 = f^{{(\rm init)}}, \quad T={epoch}$",
        scale="linear",
        **plot_kwargs,
    )
