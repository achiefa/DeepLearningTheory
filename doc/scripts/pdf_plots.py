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


def load_data(evolution):
    # Load the data used to fit the replicas
    data_by_replica_original = Distribution("Original replicas of the data")

    for rep in range(evolution.replicas):
        data = np.load(evolution.fit_folder / f"replica_{rep+1}" / "data.npy")
        data_by_replica_original.add(data)
    return data_by_replica_original


def plot_evolution_from_initialisation(evolution, ref_epoch=0, seed=SEED):
    """Plot the PDF comparison from a random initialised
    model using the frozen NTK.
    """
    # Generate the random initialised model
    xT3_0 = Distribution("xT3 at initialisation")

    replicas = evolution.replicas
    fk_grid = evolution.fk_grid
    metadata = evolution.metadata
    for rep in range(replicas):
        model = PDFmodel(
            dense_layer="Dense",
            input=fk_grid,
            outputs=1,
            architecture=metadata["model_info"]["architecture"],
            activations=["tanh", "tanh"],
            kernel_initializer="GlorotNormal",
            user_ki_args=None,
            seed=seed + rep,
        )

        xT3_0.add(model.predict().numpy().reshape(-1))

    # Load trained solution
    xT3_training = load_trained_model(evolution, -1)
    xT3_training.set_name(r"$\textrm{Trained solution}$")
    xT3_0.set_name(r"$\textrm{Initialisation}$")

    # Load data
    data_by_replica_original = load_data(evolution)

    # Get learning rate end total evolution time
    learning_rate = evolution.learning_rate
    total_evolution_time = evolution.epochs[-1] * learning_rate

    # Evolve the solution
    U, V = evolution.compute_evolution_operator(ref_epoch, total_evolution_time)
    xT3_t = U @ xT3_0 + V @ data_by_replica_original
    xT3_t.set_name(r"$\textrm{Analytical solution}$")

    produce_pdf_plot(
        evolution.fk_grid,
        [xT3_training, xT3_t],
        normalize_to=1,
        filename=f"pdf_plot_init.pdf",
        title=rf"$T_{{\rm ref}} = {{{ref_epoch}}}, \quad f_0 = f^{{(\rm init)}}$",
    )
    produce_distance_plot(
        evolution.fk_grid,
        [xT3_training, xT3_t],
        normalize_to=1,
        filename=f"distance_plot_init.pdf",
        title=rf"$T_{{\rm ref}} = {{{ref_epoch}}}, \quad f_0 = f^{{(\rm init)}}$",
    )


def plot_evolution_from_ref(evolution, ref_epoch=0):
    """Plot the PDF comparison from a random initialised
    model using the frozen NTK.
    """

    common_epochs = evolution.epochs

    # Load trained solution (end of training)
    xT3_training = load_trained_model(evolution, -1)
    xT3_training.set_name(r"$\textrm{Trained solution}$")

    # Load trained solution (at reference epoch)
    xT3_ref = load_trained_model(evolution, ref_epoch)
    xT3_ref.set_name(r"$\textrm{Trained solution at reference epoch}$")

    # Load data
    data_by_replica_original = load_data(evolution)

    # Get learning rate end total evolution time
    learning_rate = evolution.learning_rate
    t = (common_epochs[-1] - ref_epoch) * learning_rate

    # Evolve the solution
    U, V = evolution.compute_evolution_operator(ref_epoch, t)
    xT3_t = U @ xT3_ref + V @ data_by_replica_original
    xT3_t.set_name(r"$\textrm{Analytical solution}$")

    produce_pdf_plot(
        evolution.fk_grid,
        [xT3_training, xT3_t, xT3_ref],
        normalize_to=1,
        filename=f"pdf_plot_ref_{ref_epoch}.pdf",
        title=rf"$T_{{\rm ref}} = {{{ref_epoch}}}, \quad f_0 = f^{{(\rm trained)}}_{{\rm ref}}$",
    )
    produce_distance_plot(
        evolution.fk_grid,
        [xT3_training, xT3_t, xT3_ref],
        normalize_to=1,
        filename=f"distance_plot_ref_{ref_epoch}.pdf",
        title=rf"$T_{{\rm ref}} = {{{ref_epoch}}}, \quad f_0 = f^{{(\rm trained)}}_{{\rm ref}}$",
    )


def main():
    parser = ArgumentParser(description="Compute and plot PDFs.")
    parser.add_argument(
        "fitname",
        type=str,
        help="Name of the fit to compute the evolution operator for.",
    )
    fitname = parser.parse_args().fitname

    # Compute evolution operators U and V
    evolution = EvolutionOperatorComputer(fitname)

    plot_evolution_from_initialisation(evolution, ref_epoch=2000)
    plot_evolution_from_ref(evolution, ref_epoch=20000)
    plot_evolution_from_ref(evolution, ref_epoch=0)


if __name__ == "__main__":
    main()
