from pathlib import Path
import pickle

import numpy as np
import tensorflow as tf
import yaml

from yadlt.distribution import Distribution
from yadlt.model import PDFmodel

FIT_FOLDER = Path("../Results/fits/")


class EvolutionOperatorComputer:
    """
    A class to compute evolution operators U and V for different fits.

    This class handles the loading of serialized NTK data and provides
    methods to compute evolution operators at different epochs and times.
    """

    def __init__(self, fitname):
        """
        Initialize the computer with a specific fit folder.

        Parameters:
        -----------
        fit_folder : str or Path
            Path to the fit folder containing serialization data
        """
        self.fitname = fitname
        self.fit_folder = FIT_FOLDER / self.fitname
        # Ensure the fit folder exists
        if not self.fit_folder.exists():
            raise ValueError(f"Fit folder {self.fit_folder} does not exist.")
        self.serialization_folder = self.fit_folder / "serialization"

        # Load essential data
        self._load_data()
        self._load_bcdms_data()

    def _load_data(self):
        """Load serialized data from disk."""
        try:
            # Load all necessary serialized data
            self.NTK_time = pickle.load(
                open(self.serialization_folder / "NTK_time.pickle", "rb")
            )
            self.eigvals_time = pickle.load(
                open(self.serialization_folder / "eigvals_time.pickle", "rb")
            )
            self.eigvecs_time = pickle.load(
                open(self.serialization_folder / "eigvecs_time.pickle", "rb")
            )
            self.Q_by_epoch = pickle.load(
                open(self.serialization_folder / "Q_by_epoch.pickle", "rb")
            )
            self.Qinv_by_epoch = pickle.load(
                open(self.serialization_folder / "Qinv_by_epoch.pickle", "rb")
            )
            self.h_by_epoch = pickle.load(
                open(self.serialization_folder / "h_by_epoch.pickle", "rb")
            )
            self.hinv_by_epoch = pickle.load(
                open(self.serialization_folder / "hinv_by_epoch.pickle", "rb")
            )
            self.P_parallel_by_epoch = pickle.load(
                open(self.serialization_folder / "P_parallel_by_epoch.pickle", "rb")
            )
            self.P_perp_by_epoch = pickle.load(
                open(self.serialization_folder / "P_perp_by_epoch.pickle", "rb")
            )
            self.cut_by_epoch = pickle.load(
                open(self.serialization_folder / "cut.pickle", "rb")
            )
            self.common_epochs = pickle.load(
                open(self.serialization_folder / "common_epochs.pickle", "rb")
            )

            # Load metadata for learning rate
            try:
                with open(self.fit_folder / "metadata.yaml", "r") as f:
                    self._metadata = yaml.safe_load(f)
            except FileNotFoundError:
                print("Metadata file not found. Using default learning rate.")
                self._metadata = {"arguments": {"learning_rate": 1.0e-5}}

            # Load number of replicas
            replicas_folders = [
                f
                for f in self.fit_folder.iterdir()
                if f.is_dir() and "replica" in str(f)
            ]
            replicas_folders.sort()
            self.replicas = len(replicas_folders)

        except FileNotFoundError as e:
            raise RuntimeError(
                f"Could not load serialized data from {self.serialization_folder}. "
                f"Make sure the data has been serialized first. Error: {e}"
            )

    def _load_bcdms_data(self):
        """Load BCDMS data required for evolution operator computation."""
        import importlib.resources as pkg_resources

        from yadlt import data

        data_path = Path(pkg_resources.files(data) / "BCDMS_data")
        self.fk_grid = np.load(data_path / "fk_grid.npy")
        self.FK = np.load(data_path / "FK.npy")
        self.f_bcdms = np.load(data_path / "f_bcdms.npy")
        self.Cy = np.load(data_path / "Cy.npy")
        self.Cinv = np.linalg.inv(self.Cy)
        self.M = self.FK.T @ self.Cinv @ self.FK

    def compute_evolution_operator(self, reference_epoch, t):
        """
        Computes the evolution operator U(t) based on equation 41 in the solution.

        Parameters:
        -----------
        reference_epoch : int
            The reference epoch for which to compute the evolution
        t : float
            Time parameter for evolution

        Returns:
        --------
        U : Distribution
            Evolution operator U at time t
        V : Distribution
            Second operator V needed for the full solution
        """
        # Extract index of the reference epoch
        if reference_epoch not in self.common_epochs:
            raise ValueError(
                f"Reference epoch {reference_epoch} not in common epochs. "
                f"Available epochs: {self.common_epochs}"
            )

        epoch_index = self.common_epochs.index(reference_epoch)

        Q = self.Q_by_epoch[epoch_index]
        Qinv = self.Qinv_by_epoch[epoch_index]
        P_parallel = self.P_parallel_by_epoch[epoch_index]
        h = self.h_by_epoch[epoch_index]
        hinv = self.hinv_by_epoch[epoch_index].make_diagonal()

        Qt = Q.transpose()
        Qtilde = Qt @ self.M @ P_parallel
        T_tilde = Qt @ self.FK.T @ self.Cinv

        exp_ht = h.apply_operator(
            b=t,
            operator=lambda a, b: np.exp(-a * b),
            axis=0,
            name=f"exp(-h*t) at t={t}",
        ).make_diagonal()
        one_minus_exp = h.apply_operator(
            b=t,
            operator=lambda a, b: 1.0 - np.exp(-a * b),
            axis=0,
            name=f"1-exp(-h*t) at t={t}",
        ).make_diagonal()

        U_hat = Q @ exp_ht @ Qinv
        U_check = Q @ hinv @ one_minus_exp @ Qtilde
        V = Q @ hinv @ one_minus_exp @ T_tilde

        U = U_hat + U_check + P_parallel

        return U, V

    def compute_evolution_operator_at_inf(self, reference_epoch):
        """
        Computes operators at infinity (t -> inf limit).

        Parameters:
        -----------
        reference_epoch : int
            The reference epoch for which to compute the evolution

        Returns:
        --------
        U : Distribution
            Evolution operator U at infinity
        V : Distribution
            Second operator V at infinity
        """
        # Extract index of the reference epoch
        if reference_epoch not in self.common_epochs:
            raise ValueError(
                f"Reference epoch {reference_epoch} not in common epochs. "
                f"Available epochs: {self.common_epochs}"
            )

        epoch_index = self.common_epochs.index(reference_epoch)

        Q = self.Q_by_epoch[epoch_index]
        P_parallel = self.P_parallel_by_epoch[epoch_index]
        hinv = self.hinv_by_epoch[epoch_index].make_diagonal()

        Qt = Q.transpose()
        Qtilde = Qt @ self.M @ P_parallel
        T_tilde = Qt @ self.FK.T @ self.Cinv

        U_check = Q @ hinv @ Qtilde
        V = Q @ hinv @ T_tilde

        U = U_check + P_parallel

        return U, V

    @property
    def epochs(self):
        """Return list of available reference epochs."""
        return self.common_epochs

    @property
    def learning_rate(self):
        """Return the learning rate for this fit."""
        return self._metadata["arguments"]["learning_rate"]

    @property
    def metadata(self):
        """Return the metadata for this fit."""
        return self._metadata


def compute_evolution_operator(fitname, reference_epoch, t):
    """
    Compute evolution operators U and V for a given fit.

    Parameters:
    -----------
    fitname : str or Path
        Name or path of the fit folder
    reference_epoch : int
        The reference epoch for which to compute the evolution
    t : float
        Time parameter for evolution

    Returns:
    --------
    U : Distribution
        Evolution operator U at time t
    V : Distribution
        Second operator V needed for the full solution

    Example:
    --------
    >>> U, V = compute_evolution_operator('../Results/fits/250604-ac-03-L2', 20000, 0.0)
    """
    computer = EvolutionOperatorComputer(fitname)
    return computer.compute_evolution_operator(reference_epoch, t)


def compute_evolution_operator_at_inf(fitname, reference_epoch):
    """
    Compute evolution operators U and V at infinity for a given fit.

    Parameters:
    -----------
    fitname : str or Path
        Name or path of the fit folder
    reference_epoch : int
        The reference epoch for which to compute the evolution

    Returns:
    --------
    U : Distribution
        Evolution operator U at infinity
    V : Distribution
        Second operator V at infinity

    Example:
    --------
    >>> U_inf, V_inf = compute_evolution_operator_at_inf('../Results/fits/250604-ac-03-L2', 30000)
    """
    computer = EvolutionOperatorComputer(fitname)
    return computer.compute_evolution_operator_at_inf(reference_epoch)
