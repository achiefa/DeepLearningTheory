"""
This script produces plots of the PDFs in different configurations,
as well as the PDF distance.
"""

from argparse import ArgumentParser
import functools
import logging

import numpy as np

from yadlt.context import FitContext
from yadlt.distribution import Distribution
from yadlt.evolution import EvolutionOperatorComputer
from yadlt.model import generate_pdf_model, load_trained_model
from yadlt.plotting import produce_distance_plot, produce_pdf_plot

SEED = 1423413
WEIGHTS_TOKEN = "weights.h5"

from yadlt.log import setup_logger

logger = setup_logger()

# If you want to see all messages, including DEBUG:
logger.setLevel(logging.INFO)


@functools.cache
def load_model(context, epoch=-1):
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
    replicas, fk_grid_tuple, architecture_tuple, seed: int = SEED
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


def xt3_from_ref(context: FitContext, ref_epoch: int = 0):
    """Utility function to compute the xT3 from a reference epoch
    of the training process using the boundary condition of the
    reference epoch."""
    # Load trained solution (at reference epoch)
    xT3_ref = load_model(context, ref_epoch)
    xT3_ref.set_name(r"$\textrm{TS @ reference epoch}$")

    data_by_replica_original = load_data(context)

    evolution = EvolutionOperatorComputer(context)

    def func(t: float):
        U, V = evolution.compute_evolution_operator(ref_epoch, t)
        xT3_t = U @ xT3_ref + V @ data_by_replica_original
        return xT3_t

    return func


def xt3_from_initialisation(context: FitContext, ref_epoch: int = 0, seed: int = SEED):
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


def plot_evolution_from_initialisation(
    context: FitContext,
    ref_epoch: 0,
    epochs: list[int] = [0],
    seed: int = SEED,
    filename: str = "init",
    show_true: bool = False,
):
    """Plot the PDF comparison from a random initialised
    model using the frozen NTK.
    """
    learning_rate = float(context.get_config("metadata", "arguments")["learning_rate"])
    datatype = context.get_config("metadata", "arguments")["data"]
    common_epochs = context.get_config("replicas", "common_epochs")
    f_bcdms = context.load_f_bcdms()
    fk_grid = context.load_fk_grid()

    # Load trained solution
    xT3_training = load_model(context, -1)
    xT3_training.set_name(r"$\textrm{TS}$")

    # Prepare evolution from initialisation and NTK frozen
    xT3_t = xt3_from_initialisation(context, ref_epoch=ref_epoch, seed=seed)

    # Evolve the solution for each epoc
    grids_list = []
    for epoch in epochs:
        if epoch == -1:
            epoch = common_epochs[-1]

        evolution_time = epoch * learning_rate
        tmp = xT3_t(evolution_time)
        tmp.set_name(rf"$\textrm{{AS @ }} T =  {{{epoch}}}$")
        grids_list.append(tmp)

    if show_true:
        add_grid_dict = {
            "mean": f_bcdms,
            "spec": {
                "linestyle": "--",
                "label": r"$\textrm{True function}$",
                "color": "black",
            },
        }

    ax_specs_ratio = {"set_ylim": (0.5, 1.5)}

    produce_pdf_plot(
        fk_grid,
        [xT3_training, *grids_list],
        normalize_to=1,
        filename=f"pdf_plot_{filename}_{datatype}.pdf",
        title=rf"$T_{{\rm ref}} = {{{ref_epoch}}}, \quad f_0 = f^{{(\rm init)}}$",
        additional_grids=[add_grid_dict] if show_true else None,
        ax_specs=[None, ax_specs_ratio],
    )


def plot_evolution_from_ref(evolution, ref_epoch: int = 0):
    """Plot the PDF comparison from a random initialised
    model using the frozen NTK.
    """

    common_epochs = evolution.epochs
    metadata = evolution.metadata
    datatype = metadata["arguments"]["data"]

    # Load trained solution (end of training)
    xT3_training = load_model(evolution, -1)
    xT3_training.set_name(r"$\textrm{TS}$")

    # Load data
    data_by_replica_original = load_data(evolution)
    # Get learning rate end total evolution time
    learning_rate = evolution.learning_rate

    # Load trained solution (at reference epoch)
    xT3_ref = load_model(evolution, ref_epoch)
    xT3_ref.set_name(r"$\textrm{TS @ reference epoch}$")

    t = (common_epochs[-1] - ref_epoch) * learning_rate

    xT3_ref_t = xt3_from_ref(evolution, ref_epoch=ref_epoch)
    xT3_t = xT3_ref_t(t)
    xT3_t.set_name(r"$\textrm{AS}$")

    produce_pdf_plot(
        evolution.fk_grid,
        [xT3_training, xT3_t, xT3_ref],
        normalize_to=1,
        filename=f"pdf_plot_ref_{ref_epoch}_{datatype}.pdf",
        title=rf"$T_{{\rm ref}} = {{{ref_epoch}}}, \quad f_0 = f^{{(\rm trained)}}_{{T_{{\rm ref}}}}$",
    )


def plot_distance(context: FitContext, ref_epoch: int = 0, seed: int = SEED):
    """Produce a distance plot wrt the trained solution."""
    metadata = context.get_config("metadata")
    datatype = metadata["arguments"]["data"]
    learning_rate = float(context.get_config("metadata", "arguments")["learning_rate"])
    last_epoch = context.get_config("replicas", "common_epochs")[-1]
    fk_grid = context.load_fk_grid()

    # Load trained solution at the end of training
    xT3_training = load_model(context, -1)
    xT3_training.set_name(r"$\textrm{TS}$")

    # Prepare evolution from initialisation and NTK frozen
    xT3_init_t = xt3_from_initialisation(context, ref_epoch=ref_epoch, seed=seed)

    # Prepare evolution from reference epoch
    xT3_ref_t = xt3_from_ref(context, ref_epoch=0)

    # Prepare the grids
    xT3_baseline = xT3_init_t(last_epoch * learning_rate)
    xT3_baseline.set_name(r"$\textrm{AS @ last epoch}$")
    xT3_bl_epochs = xT3_init_t(5000 * learning_rate)
    xT3_bl_epochs.set_name(r"$\textrm{AS @ 5000 epochs}$")
    xT3_lazy = xT3_ref_t(last_epoch * learning_rate)
    xT3_lazy.set_name(r"$\textrm{AS lazy}$")

    produce_distance_plot(
        fk_grid,
        [xT3_training, xT3_baseline, xT3_bl_epochs, xT3_lazy],
        normalize_to=1,
        filename=f"distance_plot_{datatype}.pdf",
        title="",
        ylabel=r"$\textrm{Distance from TS}$",
        show_std=True,
        figsize=(10, 8),
    )


def plot_u_v_contributions(
    context: FitContext, ref_epoch: int = 0, ev_epoch: int = 1000, seed: int = SEED
):
    """Plot the contributions of U and V to the total operator."""
    datatype = context.get_config("metadata", "arguments")["data"]
    nreplicas = context.get_property("nreplicas")
    architecture = tuple(context.get_config("metadata", "model_info")["architecture"])
    fk_grid = context.load_fk_grid()

    # Load trained solution (end of training)
    xT3_training = load_model(context, -1)
    xT3_training.set_name(r"$\textrm{TS}$")

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

    produce_pdf_plot(
        fk_grid,
        [xT3_training, xT3_t_u, xT3_t_v],
        normalize_to=1,
        filename=f"pdf_plot_u_v_{ev_epoch}_{datatype}.pdf",
        title=rf"$T_{{\rm ref}} = {{{ref_epoch}}}, \quad f_0 = f^{{(\rm init)}}, \quad T = {{{ev_epoch}}}$",
    )


def main():
    parser = ArgumentParser(description="Compute and plot PDFs.")
    parser.add_argument(
        "fitname",
        type=str,
        help="Name of the fit to compute the evolution operator for.",
    )
    parser.add_argument(
        "--ref-epoch",
        type=int,
        default=20000,
        help="Reference epoch for the evolution operator.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Directory to save the plots. If not specified, uses the default plot directory.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force serialization of the data even if it already exists.",
    )
    fitname = parser.parse_args().fitname
    ref_epoch = parser.parse_args().ref_epoch
    plot_dir = parser.parse_args().plot_dir
    if plot_dir is not None:
        from yadlt.plotting import set_plot_dir

        set_plot_dir(plot_dir)

    context = FitContext(fitname, force_serialize=parser.parse_args().force)

    # Evolution with random initialisation, at different epochs
    plot_evolution_from_initialisation(
        context,
        ref_epoch=ref_epoch,
        epochs=[700, 5000, 50000],
        filename="init_epochs",
    )
    # Evolution with random initialisation, at the last epoch
    plot_evolution_from_initialisation(
        context,
        ref_epoch=ref_epoch,
        epochs=[-1],
        filename="init_last_epoch",
        show_true=True,
    )

    plot_u_v_contributions(context, ref_epoch=ref_epoch, ev_epoch=100, seed=SEED)
    plot_u_v_contributions(context, ref_epoch=ref_epoch, ev_epoch=50000, seed=SEED)

    plot_distance(context, ref_epoch=ref_epoch, seed=SEED)


if __name__ == "__main__":
    main()
