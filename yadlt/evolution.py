import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

from yadlt.model import compute_ntk_static

logger = logging.getLogger(__name__)

MODULE_DIR = Path(__file__).parent
FIT_FOLDER = (MODULE_DIR / "../Results/fits").resolve()


class EvolutionOperatorComputer:
    """
    A class to compute evolution operators U and V for different fits.

    This class handles the loading of serialized NTK data and provides
    methods to compute evolution operators at different epochs and times.
    """

    def __init__(self, context):
        """
        Initialize the computer with a specific fit folder.

        Parameters:
        -----------
        fit_folder : str or Path
            Path to the fit folder containing serialization data
        """
        self.context = context

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
        if reference_epoch not in self.context.common_epochs:
            raise ValueError(
                f"Reference epoch {reference_epoch} not in common epochs. "
                f"Available epochs: {self.context.common_epochs}"
            )

        epoch_index = self.context.common_epochs.index(reference_epoch)
        M = self.context.get_M()
        Cy = self.context.load_bcdms_cov()
        Cinv = np.linalg.inv(Cy)
        FK = self.context.load_fk_table()

        Q = self.context.Q_by_epoch[epoch_index]
        Qinv = self.context.Qinv_by_epoch[epoch_index]
        P_parallel = self.context.P_parallel_by_epoch[epoch_index]
        h = self.context.h_by_epoch[epoch_index]
        hinv = self.context.hinv_by_epoch[epoch_index].make_diagonal()

        Qt = Q.transpose()
        Qtilde = Qt @ M @ P_parallel
        T_tilde = Qt @ FK.T @ Cinv

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

    def compute_U_check(self, reference_epoch, t):
        """
        Computes the U_check operator for a given reference epoch and time t.

        Parameters:
        -----------
        reference_epoch : int
            The reference epoch for which to compute the U_check operator
        t : float
            Time parameter for evolution

        Returns:
        --------
        U_check : Distribution
            The U_check operator at the specified epoch and time
        """
        # Extract index of the reference epoch
        if reference_epoch not in self.context.common_epochs:
            raise ValueError(
                f"Reference epoch {reference_epoch} not in common epochs. "
                f"Available epochs: {self.context.common_epochs}"
            )

        epoch_index = self.context.common_epochs.index(reference_epoch)
        M = self.context.get_M()

        Q = self.context.Q_by_epoch[epoch_index]
        P_parallel = self.context.P_parallel_by_epoch[epoch_index]
        h = self.context.h_by_epoch[epoch_index]
        hinv = self.context.hinv_by_epoch[epoch_index].make_diagonal()

        Qt = Q.transpose()
        Qtilde = Qt @ M @ P_parallel

        one_minus_exp = h.apply_operator(
            b=t,
            operator=lambda a, b: 1.0 - np.exp(-a * b),
            axis=0,
            name=f"1-exp(-h*t) at t={t}",
        ).make_diagonal()

        U_check = Q @ hinv @ one_minus_exp @ Qtilde

        return U_check

    def compute_M_operator(self, reference_epoch, t):
        """
        Computes the matrix M that enters the evolution operator
        """
        epoch_index = self.context.common_epochs.index(reference_epoch)
        Q = self.context.Q_by_epoch[epoch_index]
        h = self.context.h_by_epoch[epoch_index]
        hinv = self.context.hinv_by_epoch[epoch_index].make_diagonal()

        Qt = Q.transpose()
        one_minus_exp = h.apply_operator(
            b=t,
            operator=lambda a, b: 1.0 - np.exp(-a * b),
            axis=0,
            name=f"1-exp(-h*t) at t={t}",
        ).make_diagonal()

        Mcal = Q @ hinv @ one_minus_exp @ Qt
        return Mcal

    def get_P_parallel(self, reference_epoch):
        """
        Returns the parallel projector P_parallel for a given reference epoch.

        Parameters:
        -----------
        reference_epoch : int
            The reference epoch for which to get the parallel projector

        Returns:
        --------
        P_parallel : Distribution
            The parallel projector at the specified epoch
        """
        epoch_index = self.context.common_epochs.index(reference_epoch)
        return self.context.P_parallel_by_epoch[epoch_index]

    def get_Z(self, reference_epoch):
        """
        Returns the Z matrix for a given reference epoch.

        Parameters:
        -----------
        reference_epoch : int
            The reference epoch for which to get the Z matrix

        Returns:
        --------
        Z : Distribution
            The Z matrix at the specified epoch
        """
        epoch_index = self.context.common_epochs.index(reference_epoch)
        return self.context.eigvecs_time[epoch_index]

    def get_cut(self, reference_epoch):
        """
        Returns the cut index for a given reference epoch.

        Parameters:
        -----------
        reference_epoch : int
            The reference epoch for which to get the cut index

        Returns:
        --------
        cut : int
            The cut index at the specified epoch
        """
        epoch_index = self.context.common_epochs.index(reference_epoch)
        return self.context.cut_by_epoch[epoch_index]

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
        if reference_epoch not in self.context.common_epochs:
            raise ValueError(
                f"Reference epoch {reference_epoch} not in common epochs. "
                f"Available epochs: {self.context.common_epochs}"
            )

        epoch_index = self.context.common_epochs.index(reference_epoch)
        M = self.context.get_M()
        Cy = self.context.load_bcdms_cov()
        Cinv = np.linalg.inv(Cy)
        FK = self.context.load_fk_table()

        Q = self.context.Q_by_epoch[epoch_index]
        P_parallel = self.context.P_parallel_by_epoch[epoch_index]
        hinv = self.context.hinv_by_epoch[epoch_index].make_diagonal()

        Qt = Q.transpose()
        Qtilde = Qt @ M @ P_parallel
        T_tilde = Qt @ FK.T @ Cinv

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


def process_model(model, grid, M):
    """
    Helper function to process a single replica at a given epoch.
    """
    # Compute the NTK for the current replica and epoch
    ntk = compute_ntk_static(grid, model, 1)
    size = ntk.shape[0]

    # Compute eigenvalues and eigenvectors of the NTK
    Z, eigenvalues, ZrT = np.linalg.svd(ntk, hermitian=True)

    # Compute frobenius norm
    frob_norm = np.sqrt(np.sum([s**2 for s in eigenvalues]))

    for idx in range(len(eigenvalues)):
        if not np.allclose(Z[:, idx], ZrT.T[:, idx]):
            cut = idx
            break

    for i in range(cut, 0, -1):
        if eigenvalues[i] / eigenvalues[0] > 1.0e-7:
            cut = np.int64(i + 1)
            break

    perp_mask = [True] * cut + [False] * (size - cut)
    parallel_mask = ~np.array(perp_mask)

    Lambda_perp = eigenvalues[perp_mask]
    Z_perp = Z[:, perp_mask]
    Z_parallel = Z[:, parallel_mask]

    # Parallel projector
    P_parallel = np.empty((size, size))
    P_perp = np.empty((size, size))
    P_parallel = np.dot(Z_parallel, Z_parallel.T)
    P_perp = np.dot(Z_perp, Z_perp.T)

    # Compute similarity transformation
    Lambda_perp_sqrt = np.sqrt(Lambda_perp)
    Lambda_perp_sqrt_inv = 1.0 / Lambda_perp_sqrt
    P = np.diag(Lambda_perp_sqrt_inv) @ Z_perp.T
    P_inv = Z_perp @ np.diag(Lambda_perp_sqrt)

    # Symmetric operator
    H_perp = (
        np.diag(Lambda_perp_sqrt) @ Z_perp.T @ M @ Z_perp @ np.diag(Lambda_perp_sqrt)
    )

    # Eigendecomposition of H_perp
    h, W = np.linalg.eigh(H_perp)
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(h)[::-1]
    h, W = h[idx], W[:, idx]

    hinv = 1 / h

    # Compute Q and its inverse
    Q = P_inv @ W
    Qinv = W.T @ P

    # Pad quantities to ensure they are square matrices
    Q = np.pad(
        Q, ((0, 0), (0, Q.shape[0] - Q.shape[1])), mode="constant", constant_values=0
    )
    Qinv = np.pad(
        Qinv,
        ((0, Qinv.shape[1] - Qinv.shape[0]), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    hinv = np.pad(
        hinv, (0, Q.shape[0] - hinv.shape[0]), mode="constant", constant_values=0
    )
    h = np.pad(h, (0, Q.shape[0] - h.shape[0]), mode="constant", constant_values=0)

    return {
        "NTK": ntk,
        "Z": Z,
        "frob_norm": frob_norm,
        "eigenvalues": eigenvalues,
        "P_parallel": P_parallel,
        "P_perp": P_perp,
        "Q": Q,
        "Qinv": Qinv,
        "h": h,
        "hinv": hinv,
        "cut": cut,
    }
