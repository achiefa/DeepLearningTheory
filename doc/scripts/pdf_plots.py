"""
This script produces plots of the PDFs in different configurations,
as well as the PDF distance.
"""

from argparse import ArgumentParser
import functools

import numpy as np

from yadlt.distribution import Distribution
from yadlt.evolution import EvolutionOperatorComputer
from yadlt.model import PDFmodel
from yadlt.plotting import produce_distance_plot, produce_pdf_plot

SEED = 1423413
WEIGHTS_TOKEN = "weights.h5"


@functools.cache
def load_trained_model(evolution, epoch=-1):
    # Extract the last common epoch from the ensemble of replicas
    common_epochs = evolution.epochs
    epoch_idx = -1 if epoch == -1 else common_epochs.index(epoch)

    # Extract replicas from the fit folder
    replicas_folders = [
        f for f in evolution.fit_folder.iterdir() if f.is_dir() and "replica" in str(f)
    ]
    replicas_folders.sort()

    replica_epochs_dict = {}
    for replica_folder in replicas_folders:
        epochs = [
            f
            for f in replica_folder.iterdir()
            if f.is_file() and WEIGHTS_TOKEN in str(f)
        ]
        epochs.sort()
        replica_epochs_dict[replica_folder.name] = epochs

    xT3_training = Distribution("xT3_training")

    for replica_path in replicas_folders:
        replica = replica_path.name
        epoch = replica_epochs_dict[replica][epoch_idx]
        model = PDFmodel.load_model(replica_path / "config.json", epoch)
        xT3_training.add(model.predict().numpy().reshape(-1))

    return xT3_training


@functools.cache
def load_data(evolution):
    # Load the data used to fit the replicas
    data_by_replica_original = Distribution("Original replicas of the data")

    for rep in range(evolution.replicas):
        data = np.load(evolution.fit_folder / f"replica_{rep+1}" / "data.npy")
        data_by_replica_original.add(data)
    return data_by_replica_original


@functools.cache
def produce_model_at_initialisation(
    replicas, fk_grid_tuple, architecture_tuple, seed: int = SEED
):
    """Initialise a model at random initialisation."""
    xT3_0 = Distribution("xT3 at initialisation")
    for rep in range(replicas):
        model = PDFmodel(
            dense_layer="Dense",
            input=np.array(fk_grid_tuple),
            outputs=1,
            architecture=list(architecture_tuple),
            activations=["tanh", "tanh"],
            kernel_initializer="GlorotNormal",
            user_ki_args=None,
            seed=seed + rep,
        )

        xT3_0.add(model.predict().numpy().reshape(-1))

    return xT3_0


def xt3_from_ref(evolution: EvolutionOperatorComputer, ref_epoch: int = 0):
    """Utility function to compute the xT3 from a reference epoch
    of the training process using the boundary condition of the
    reference epoch."""
    # Load trained solution (at reference epoch)
    xT3_ref = load_trained_model(evolution, ref_epoch)
    xT3_ref.set_name(r"$\textrm{TS @ reference epoch}$")

    data_by_replica_original = load_data(evolution)

    def func(t: float):
        U, V = evolution.compute_evolution_operator(ref_epoch, t)
        xT3_t = U @ xT3_ref + V @ data_by_replica_original
        return xT3_t

    return func


def xt3_from_initialisation(
    evolution: EvolutionOperatorComputer, ref_epoch: int = 0, seed: int = SEED
):
    """Utility function to compute the xT3 from a random initialisation
    using a frozen NTK specified by the reference epoch."""
    replicas = evolution.replicas
    fk_grid = evolution.fk_grid
    metadata = evolution.metadata
    arch_tuple = tuple(metadata["model_info"]["architecture"])

    xT3_0 = produce_model_at_initialisation(
        replicas=replicas,
        fk_grid_tuple=tuple(fk_grid),
        architecture_tuple=arch_tuple,
        seed=seed,
    )
    data_by_replica_original = load_data(evolution)

    def func(t: float):
        # Load data
        U, V = evolution.compute_evolution_operator(ref_epoch, t)
        xT3_t = U @ xT3_0 + V @ data_by_replica_original
        return xT3_t

    return func


def plot_evolution_from_initialisation(
    evolution: EvolutionOperatorComputer,
    ref_epoch: 0,
    epochs: list[int] = [0],
    seed: int = SEED,
    filename: str = "init",
    show_t3_true: bool = False,
):
    """Plot the PDF comparison from a random initialised
    model using the frozen NTK.
    """
    learning_rate = evolution.learning_rate
    metadata = evolution.metadata
    datatype = metadata["arguments"]["data"]

    # Load trained solution
    xT3_training = load_trained_model(evolution, -1)
    xT3_training.set_name(r"$\textrm{TS}$")

    # Prepare evolution from initialisation and NTK frozen
    xT3_t = xt3_from_initialisation(evolution, ref_epoch=ref_epoch, seed=seed)

    # Evolve the solution for each epoc
    grids_list = []
    for epoch in epochs:
        if epoch == -1:
            epoch = evolution.epochs[-1]

        evolution_time = epoch * learning_rate
        tmp = xT3_t(evolution_time)
        tmp.set_name(rf"$\textrm{{AS @ }} T =  {{{epoch}}}$")
        grids_list.append(tmp)

    if show_t3_true:
        t3 = evolution.f_bcdms

    produce_pdf_plot(
        evolution.fk_grid,
        [xT3_training, *grids_list],
        normalize_to=1,
        filename=f"pdf_plot_{filename}_{datatype}.pdf",
        title=rf"$T_{{\rm ref}} = {{{ref_epoch}}}, \quad f_0 = f^{{(\rm init)}}$",
    )


def plot_evolution_from_ref(evolution, ref_epoch: int = 0):
    """Plot the PDF comparison from a random initialised
    model using the frozen NTK.
    """

    common_epochs = evolution.epochs
    metadata = evolution.metadata
    datatype = metadata["arguments"]["data"]

    # Load trained solution (end of training)
    xT3_training = load_trained_model(evolution, -1)
    xT3_training.set_name(r"$\textrm{TS}$")

    # Load data
    data_by_replica_original = load_data(evolution)
    # Get learning rate end total evolution time
    learning_rate = evolution.learning_rate

    # Load trained solution (at reference epoch)
    xT3_ref = load_trained_model(evolution, ref_epoch)
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


def plot_distance(
    evolution: EvolutionOperatorComputer, ref_epoch: int = 0, seed: int = SEED
):
    """Produce a distance plot wrt the trained solution."""
    metadata = evolution.metadata
    datatype = metadata["arguments"]["data"]
    learning_rate = evolution.learning_rate
    last_epoch = evolution.epochs[-1]

    # Load trained solution at the end of training
    xT3_training = load_trained_model(evolution, -1)
    xT3_training.set_name(r"$\textrm{TS}$")

    # Prepare evolution from initialisation and NTK frozen
    xT3_init_t = xt3_from_initialisation(evolution, ref_epoch=ref_epoch, seed=seed)

    # Prepare evolution from reference epoch
    xT3_ref_t = xt3_from_ref(evolution, ref_epoch=0)

    # Prepare the grids
    xT3_baseline = xT3_init_t(last_epoch * learning_rate)
    xT3_baseline.set_name(r"$\textrm{AS @ last epoch}$")
    xT3_bl_epochs = xT3_init_t(5000 * learning_rate)
    xT3_bl_epochs.set_name(r"$\textrm{AS @ 5000 epochs}$")
    xT3_lazy = xT3_ref_t(last_epoch * learning_rate)
    xT3_lazy.set_name(r"$\textrm{AS lazy}$")

    produce_distance_plot(
        evolution.fk_grid,
        [xT3_training, xT3_baseline, xT3_bl_epochs, xT3_lazy],
        normalize_to=1,
        filename=f"distance_plot_{datatype}.pdf",
        title="",
        ylabel=r"$\textrm{Distance from TS}$",
        show_std=True,
        figsize=(10, 8),
    )


def plot_u_v_contributions(evolution, ref_epoch: int = 0, ev_epoch: int = 1000):
    """Plot the contributions of U and V to the total operator."""
    metadata = evolution.metadata
    datatype = metadata["arguments"]["data"]

    # Load trained solution (end of training)
    xT3_training = load_trained_model(evolution, -1)
    xT3_training.set_name(r"$\textrm{TS}$")

    # Load data
    data_by_replica_original = load_data(evolution)

    # Get learning rate
    learning_rate = evolution.learning_rate
    t = ev_epoch * learning_rate

    # Load trained solution (at reference epoch)
    xT3_ref = load_trained_model(evolution, ref_epoch)
    xT3_ref.set_name(rf"$\textrm{{TS @ }} T_{{\rm ref}}$")

    # Evolve the solution
    U, V = evolution.compute_evolution_operator(ref_epoch, t)
    xT3_t_u = U @ xT3_ref
    xT3_t_u.set_name(r"$\textrm{Contribution from U}$")
    xT3_t_v = V @ data_by_replica_original
    xT3_t_v.set_name(r"$\textrm{Contribution from V}$")

    produce_pdf_plot(
        evolution.fk_grid,
        [xT3_training, xT3_t_u, xT3_t_v],
        normalize_to=1,
        filename=f"pdf_plot_u_v_{ev_epoch}_{datatype}.pdf",
        title=rf"$T_{{\rm ref}} = {{{ref_epoch}}}, \quad f_0 = f^{{(\rm trained)}}_{{T_{{\rm ref}}}}, \quad T = {{{ev_epoch}}}$",
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
    fitname = parser.parse_args().fitname
    ref_epoch = parser.parse_args().ref_epoch
    plot_dir = parser.parse_args().plot_dir
    if plot_dir is not None:
        from yadlt.plotting import set_plot_dir

        set_plot_dir(plot_dir)

    # Compute evolution operators U and V
    evolution = EvolutionOperatorComputer(fitname)

    plot_evolution_from_initialisation(
        evolution,
        ref_epoch=ref_epoch,
        epochs=[700, 5000, 100000],
        filename="init_epochs",
    )
    plot_evolution_from_initialisation(
        evolution, ref_epoch=ref_epoch, epochs=[-1], filename="init_last_epoch"
    )

    # plot_evolution_from_ref(evolution, ref_epoch=ref_epoch)
    # plot_evolution_from_ref(evolution, ref_epoch=0)

    # plot_u_v_contributions(evolution, ref_epoch=ref_epoch, ev_epoch=100)
    # plot_u_v_contributions(evolution, ref_epoch=ref_epoch, ev_epoch=50000)

    # plot_distance(evolution, ref_epoch=ref_epoch, seed=SEED)


if __name__ == "__main__":
    main()
