"""Module for loading the data required in the training"""

import functools
import importlib.resources as pkg_resources
from pathlib import Path

import numpy as np

from yadlt import data

DATA_PATH = Path(pkg_resources.files(data) / "BCDMS_data")


@functools.lru_cache(maxsize=None)
def load_bcdms_grid():
    """Load the BCDMS grid data."""
    return np.load(DATA_PATH / "fk_grid.npy")


@functools.lru_cache(maxsize=None)
def load_bcdms_pdf():
    """Load the BCDMS PDF data."""
    return np.load(DATA_PATH / "f_bcdms.npy")


@functools.lru_cache(maxsize=None)
def load_bcdms_cov():
    """Load the BCDMS covariance data."""
    return np.load(DATA_PATH / "Cy.npy")


@functools.lru_cache(maxsize=None)
def load_bcdms_fk():
    """Load the BCDMS Fourier transform data."""
    return np.load(DATA_PATH / "FK.npy")


@functools.lru_cache(maxsize=None)
def load_bcdms_data():
    """Load all BCDMS data."""
    return np.load(DATA_PATH / "data.npy")
