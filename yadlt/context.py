"""
Context class for managing global state and configuration across the application.

This class implements the singleton pattern to ensure only one instance exists
throughout the application lifecycle, providing a centralized location for
shared properties and configuration.
"""

from abc import ABC
import functools
import logging
from pathlib import Path
import pickle
import threading
from typing import Any, Dict

import numpy as np
import tensorflow as tf
import yaml

from yadlt.distribution import Distribution
from yadlt.evolution import process_model
from yadlt.load_data import (
    load_bcdms_cov,
    load_bcdms_fk,
    load_bcdms_grid,
    load_bcdms_pdf,
)
from yadlt.model import load_trained_model, load_weights

logger = logging.getLogger(__name__)

MODULE_DIR = Path(__file__).parent
FIT_FOLDER = (MODULE_DIR / "../Results/fits").resolve()


class Context(ABC):
    """
    Singleton context class for managing application-wide state and configuration.

    This class provides a centralized location for storing and accessing
    shared properties, configuration settings, and runtime state across
    the entire application.
    """

    _instances: Dict[str, "Context"] = {}
    _lock = threading.Lock()

    def __new__(cls, context_name: str, **kwargs) -> "Context":
        """
        Ensure only one instance of Context exists per config_name (multiton
        pattern). Thread-safe implementation using double-checked locking.
        """
        if context_name in cls._instances:
            return cls._instances[context_name]

        with cls._lock:
            if context_name not in cls._instances:
                instance = super(Context, cls).__new__(cls)
                instance._initialized = False
                instance._context_name = context_name
                cls._instances[context_name] = instance
            return cls._instances[context_name]

    def __init__(self, context_name: str) -> None:
        """Initialize the context instance only once."""
        if not self._initialized:
            self._initialized = True
            self._properties: Dict[str, Any] = {}
            self._config: Dict[str, Any] = {}
            self.context_name = context_name

    @classmethod
    def get_instance(cls, context_name: str) -> "Context":
        """
        Get the singleton instance of Context for the given context_name.

        Args:
            context_name: Unique name for the context instance

        Returns:
            Context instance for the specified context_name
        """
        return cls(context_name)

    # Property management methods
    def set_property(self, key: str, value: Any) -> None:
        """
        Set a property value.

        Args:
            key: Property key
            value: Property value
        """
        self._properties[key] = value
        logger.debug(f"Property '{key}' set to: {value}")

    def get_property(self, key: str, default: Any = None) -> Any:
        """
        Get a property value.

        Args:
            key: Property key
            default: Default value if key doesn't exist

        Returns:
            Property value or default
        """
        return self._properties.get(key, default)

    def has_property(self, key: str) -> bool:
        """
        Check if a property exists.

        Args:
            key: Property key

        Returns:
            True if property exists, False otherwise
        """
        return key in self._properties

    def remove_property(self, key: str) -> bool:
        """
        Remove a property.

        Args:
            key: Property key

        Returns:
            True if property was removed, False if it didn't exist
        """
        if key in self._properties:
            del self._properties[key]
            logger.debug(f"Property '{key}' removed")
            return True
        return False

    def get_all_properties(self) -> Dict[str, Any]:
        """
        Get all properties.

        Returns:
            Dictionary of all properties
        """
        return self._properties.copy()

    # Configuration management methods
    def set_config(self, section: str, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            section: Configuration section
            key: Configuration key
            value: Configuration value
        """
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value
        logger.debug(f"Config '{section}.{key}' set to: {value}")

    def get_config(self, section: str, key: str = None, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            section: Configuration section
            key: Configuration key (optional, returns entire section if None)
            default: Default value if key doesn't exist

        Returns:
            Configuration value or default
        """
        if key is None:
            return self._config.get(section, default)
        return self._config.get(section, {}).get(key, default)

    def update_config(self, section: str, config_dict: Dict[str, Any]) -> None:
        """
        Update an entire configuration section.

        Args:
            section: Configuration section
            config_dict: Dictionary of configuration values
        """
        if section not in self._config:
            self._config[section] = {}
        self._config[section].update(config_dict)
        logger.debug(f"Config section '{section}' updated")

    def get_config_keys(self):
        """
        Get all keys in the configuration.

        Returns:
            List of all configuration keys
        """
        for section in self._config.keys():
            print(f"\nSection: {section}: ")
            for key in self._config[section].keys():
                print(f"  - {key}")

    # Utility methods
    def reset(self) -> None:
        """Reset the context to default state."""
        self._properties.clear()
        self._config.clear()
        self._init_default_properties()
        logger.info("Context reset to default state")

    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """
        Load properties and configuration from a dictionary.

        Args:
            data: Dictionary containing properties and config
        """
        if "properties" in data:
            self._properties.update(data["properties"])
        if "config" in data:
            self._config.update(data["config"])
        logger.info("Context loaded from dictionary")

    def to_dict(self) -> Dict[str, Any]:
        """
        Export context to dictionary.

        Returns:
            Dictionary containing all properties and configuration
        """
        return {"properties": self._properties.copy(), "config": self._config.copy()}

    def __str__(self) -> str:
        """String representation of the context."""
        return f"Context(properties={len(self._properties)}, config_sections={len(self._config)})"

    def __repr__(self) -> str:
        """Detailed string representation of the context."""
        return f"Context(properties={self._properties}, config={self._config})"


class FitContext(Context):
    """
    Specialized context for managing fit-specific properties and configuration.

    Inherits from Context and provides additional methods for fit management.
    """

    WEIGHTS_TOKEN = "weights.h5"

    def __init__(
        self, fit_name: str, fit_folder: str = None, force_serialize: bool = False
    ) -> None:
        super().__init__(fit_name)

        logger.info(f"Initializing FitContext for fit: {fit_name}")
        self.fit_name = fit_name

        self.fit_folder = fit_folder or (FIT_FOLDER / self.fit_name)
        if not self.fit_folder.exists():
            raise ValueError(f"Fit folder {self.fit_folder} does not exist.")

        logger.info(f"FitContext initialized for fit: {self.fit_name}")

        # Initialize serialization folder if it doesn't exist
        serialization_folder = self.fit_folder / "serialization"
        if not serialization_folder.exists():
            serialization_folder.mkdir(parents=True, exist_ok=True)
        self.set_config("folders", "serialization_folder", serialization_folder)

        # Initialize plot folder if it doesn't exist
        plot_folder = self.fit_folder / "plots"
        if not plot_folder.exists():
            plot_folder.mkdir(parents=True, exist_ok=True)
        self.set_config("folders", "plot_folder", plot_folder)

        # Initialize serialization folder if it doesn't exist
        serialization_folder = self.fit_folder / "serialization"
        if not serialization_folder.exists():
            serialization_folder.mkdir(parents=True, exist_ok=True)
        self.set_config("folders", "serialization_folder", serialization_folder)

        # Load metadata
        metadata_file = self.fit_folder / "metadata.yaml"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = yaml.safe_load(f)
            self.update_config("metadata", metadata)
            logger.info("Metadata loaded from file")

        self._init_default_properties()
        self._init_replicas()
        self._init_common_epochs()
        self._load_data(force_serialize=force_serialize)

    def _init_default_properties(self):
        """Initialize default properties and configuration."""
        # Open metadata file and read existing properties
        self._properties.update(
            {
                "device": "cpu",
            }
        )

    def _init_replicas(self):
        """Initialize and manage replicas for the fit context."""
        # Extract replicas from the fit folder
        replicas_folders = [
            f for f in self.fit_folder.iterdir() if f.is_dir() and "replica" in str(f)
        ]
        replicas_folders.sort(key=lambda x: int(x.name.split("_")[-1]))
        self.set_property("nreplicas", len(replicas_folders))
        self.set_config("folders", "replicas_folders", replicas_folders)

        epochs_by_replica = {}
        # For each replicas, load the epochs and dat
        for replica_folder in replicas_folders:
            epochs = [
                f
                for f in replica_folder.iterdir()
                if f.is_file() and FitContext.WEIGHTS_TOKEN in str(f)
            ]
            epochs.sort()
            epochs_by_replica[replica_folder.name] = epochs
        self.set_config("replicas", "epochs_by_replica", epochs_by_replica)

    def _init_common_epochs(self):
        """Initialize common epochs across all replicas."""
        common_epochs = set()
        epochs_by_replica = self.get_config("replicas", "epochs_by_replica", {})
        for _, epochs in epochs_by_replica.items():
            aux = set()
            for epoch in epochs:
                epoch_num = int(epoch.name.split(".")[0].split("_")[1])
                aux.add(epoch_num)

            if len(common_epochs) == 0:
                common_epochs = aux
            else:
                common_epochs.intersection_update(aux)

        common_epochs = sorted(list(common_epochs))
        self.set_config("replicas", "common_epochs", common_epochs)

    @functools.lru_cache(maxsize=None)
    def load_fk_grid(self):
        """Load and return the FK grid data."""
        fk_grid = load_bcdms_grid()
        return fk_grid

    @functools.lru_cache(maxsize=None)
    def load_fk_table(self):
        """Load the FK table data."""
        fk_table = load_bcdms_fk()
        return fk_table

    @functools.lru_cache(maxsize=None)
    def load_f_bcdms(self):
        """Load the BCDMS PDF"""
        pdf = load_bcdms_pdf()
        return pdf

    @functools.lru_cache(maxsize=None)
    def load_bcdms_cov(self):
        """Load the BCDMS covariance matrix."""
        cov = load_bcdms_cov()
        return cov

    @functools.lru_cache(maxsize=None)
    def get_M(self):
        """Get the M matrix from the FK grid."""
        fk = self.load_fk_table()
        C = self.load_bcdms_cov()
        Cinv = np.linalg.inv(C)
        M = fk.T @ Cinv @ fk
        return M

    def get_data_by_replica(self, replica_idx: int):
        data = np.load(self.fit_folder / f"replica_{replica_idx+1}" / "data.npy")
        return data

    def _load_data(self, force_serialize=False):
        """Load serialized data from disk."""

        if force_serialize:
            logger.info("Forcing serialization of data...")
            self._serialize_data()
            return

        try:
            # Load all necessary serialized data
            serialization_folder = self.get_config("folders", "serialization_folder")
            logger.info(f"Loading serialized data from {serialization_folder}...")
            self.NTK_time = pickle.load(
                open(serialization_folder / "NTK_time.pickle", "rb")
            )
            self.frob_norm_time = pickle.load(
                open(serialization_folder / "frob_norm_time.pickle", "rb")
            )
            self.eigvals_time = pickle.load(
                open(serialization_folder / "eigvals_time.pickle", "rb")
            )
            self.eigvecs_time = pickle.load(
                open(serialization_folder / "eigvecs_time.pickle", "rb")
            )
            self.Q_by_epoch = pickle.load(
                open(serialization_folder / "Q_by_epoch.pickle", "rb")
            )
            self.Qinv_by_epoch = pickle.load(
                open(serialization_folder / "Qinv_by_epoch.pickle", "rb")
            )
            self.h_by_epoch = pickle.load(
                open(serialization_folder / "h_by_epoch.pickle", "rb")
            )
            self.hinv_by_epoch = pickle.load(
                open(serialization_folder / "hinv_by_epoch.pickle", "rb")
            )
            self.P_parallel_by_epoch = pickle.load(
                open(serialization_folder / "P_parallel_by_epoch.pickle", "rb")
            )
            self.P_perp_by_epoch = pickle.load(
                open(serialization_folder / "P_perp_by_epoch.pickle", "rb")
            )
            self.cut_by_epoch = pickle.load(
                open(serialization_folder / "cut.pickle", "rb")
            )
            self.common_epochs = pickle.load(
                open(serialization_folder / "common_epochs.pickle", "rb")
            )
            return

        except FileNotFoundError:
            logger.warning(
                f"Could not load serialized data from {serialization_folder}. "
                f"Serialization is starting..."
            )
            self._serialize_data()
            self._load_data(force_serialize=False)

    def _serialize_data(self):
        nreplicas = self.get_config("replicas", "nreplicas")
        fk_grid = self.load_fk_grid()
        common_epochs = self.get_config("replicas", "common_epochs", [])
        serialization_folder = self.get_config("folders", "serialization_folder")
        replica_folders = self.get_config("folders", "replicas_folders")
        logger.info(f"Serializing data to {serialization_folder}...")

        x = tf.constant(fk_grid.reshape(1, -1, 1), dtype=tf.float32)

        # Initialize distributions for serialization
        NTK_time = [
            Distribution(
                name=f"NTK at epoch {epoch}",
                size=nreplicas,
                shape=(
                    fk_grid.size,
                    fk_grid.size,
                ),
            )
            for epoch in common_epochs
        ]
        frob_norm_time = [
            Distribution(
                name=f"Frobenius norm at epoch {epoch}", size=nreplicas, shape=()
            )
            for epoch in common_epochs
        ]
        eigvals_time = [
            Distribution(
                name=f"Eigenvalues of the NTK at epoch {epoch}",
                size=nreplicas,
                shape=(fk_grid.size,),
            )
            for epoch in common_epochs
        ]
        eigvecs_time = [
            Distribution(
                name=f"Eigenvectors of the NTK at epoch {epoch}",
                size=nreplicas,
                shape=(fk_grid.size, fk_grid.size),
            )
            for epoch in common_epochs
        ]
        P_parallel_by_epoch = [
            Distribution(
                name=f"P at parallel epoch {epoch}",
                size=nreplicas,
                shape=(
                    fk_grid.size,
                    fk_grid.size,
                ),
            )
            for epoch in common_epochs
        ]
        P_perp_by_epoch = [
            Distribution(
                name=f"P at perpendicular epoch {epoch}",
                size=nreplicas,
                shape=(
                    fk_grid.size,
                    fk_grid.size,
                ),
            )
            for epoch in common_epochs
        ]
        Q_by_epoch = [
            Distribution(
                name=f"Q at epoch {epoch}",
                size=nreplicas,
                shape=(
                    fk_grid.size,
                    fk_grid.size,
                ),
            )
            for epoch in common_epochs
        ]
        Qinv_by_epoch = [
            Distribution(
                name=f"Q inv. at epoch {epoch}",
                size=nreplicas,
                shape=(
                    fk_grid.size,
                    fk_grid.size,
                ),
            )
            for epoch in common_epochs
        ]
        h_by_epoch = [
            Distribution(
                name=f"h at epoch {epoch}", size=nreplicas, shape=(fk_grid.size,)
            )
            for epoch in common_epochs
        ]
        hinv_by_epoch = [
            Distribution(
                name=f"h at epoch {epoch}", size=nreplicas, shape=(fk_grid.size,)
            )
            for epoch in common_epochs
        ]
        cut_by_epoch = [
            Distribution(name=f"cut at epoch {epoch}", size=nreplicas, shape=())
            for epoch in common_epochs
        ]

        # Load dummy model
        model, _ = load_trained_model(replica_folders[0], epoch=500)
        for epoch in common_epochs:
            logger.info(f"Processing epoch {epoch} / {common_epochs[-1]}")

            # Loop over each replica
            for replica_path in replica_folders:
                weight_file = load_weights(replica_path, epoch=epoch)
                try:
                    model.load_weights(weight_file)
                except ValueError as e:  # Handle legacy
                    pdf_model = model.layers[1]
                    pdf_model.load_weights(weight_file)

                result = process_model(model, x, self.get_M())

                NTK_time[common_epochs.index(epoch)].add(result["NTK"])
                frob_norm_time[common_epochs.index(epoch)].add(result["frob_norm"])
                eigvals_time[common_epochs.index(epoch)].add(result["eigenvalues"])
                eigvecs_time[common_epochs.index(epoch)].add(result["Z"])

                P_parallel_by_epoch[common_epochs.index(epoch)].add(
                    result["P_parallel"]
                )
                P_perp_by_epoch[common_epochs.index(epoch)].add(result["P_perp"])

                h_by_epoch[common_epochs.index(epoch)].add(result["h"])
                hinv_by_epoch[common_epochs.index(epoch)].add(result["hinv"])
                Q_by_epoch[common_epochs.index(epoch)].add(result["Q"])
                Qinv_by_epoch[common_epochs.index(epoch)].add(result["Qinv"])
                cut_by_epoch[common_epochs.index(epoch)].add(result["cut"])

        pickle.dump(NTK_time, open(serialization_folder / "NTK_time.pickle", "wb"))
        pickle.dump(
            frob_norm_time, open(serialization_folder / "frob_norm_time.pickle", "wb")
        )
        pickle.dump(
            eigvals_time, open(serialization_folder / "eigvals_time.pickle", "wb")
        )
        pickle.dump(
            eigvecs_time, open(serialization_folder / "eigvecs_time.pickle", "wb")
        )
        pickle.dump(Q_by_epoch, open(serialization_folder / "Q_by_epoch.pickle", "wb"))
        pickle.dump(
            Qinv_by_epoch, open(serialization_folder / "Qinv_by_epoch.pickle", "wb")
        )
        pickle.dump(h_by_epoch, open(serialization_folder / "h_by_epoch.pickle", "wb"))
        pickle.dump(
            hinv_by_epoch, open(serialization_folder / "hinv_by_epoch.pickle", "wb")
        )
        pickle.dump(
            P_parallel_by_epoch,
            open(serialization_folder / "P_parallel_by_epoch.pickle", "wb"),
        )
        pickle.dump(
            P_perp_by_epoch, open(serialization_folder / "P_perp_by_epoch.pickle", "wb")
        )
        pickle.dump(cut_by_epoch, open(serialization_folder / "cut.pickle", "wb"))
        pickle.dump(
            common_epochs, open(serialization_folder / "common_epochs.pickle", "wb")
        )

        logger.info(f"Data serialized to {serialization_folder}")
