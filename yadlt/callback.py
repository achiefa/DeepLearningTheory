import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class WeightStorageCallback(tf.keras.callbacks.Callback):
    """Callback to store model weights every f epochs"""

    def __init__(self, storage_frequency=100, storage_path=None):
        super().__init__()
        self.storage_frequency = storage_frequency
        self.storage_path = storage_path or "model_weights"
        self.epoch_checkpoints = []
        self.loss_checkpoints = []

        # Create storage directory
        Path(self.storage_path).mkdir(exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.storage_frequency == 0:
            # Store weights in memory
            self.epoch_checkpoints.append(epoch + 1)
            self.loss_checkpoints.append(logs.get("loss", 0))

            # Get PDF model weights
            pdf_model = self.model.get_layer("Observable").get_layer("PDF")

            # Save weights to file
            weight_filename = f"{self.storage_path}/epoch_{epoch + 1}.weights.h5"
            pdf_model.model.save_weights(weight_filename)

    def save_history(self, filename=None):
        """Save the complete training history to a file"""
        if filename is None:
            filename = f"{self.storage_path}/complete_training_history.npz"

        # Convert to numpy arrays for storage
        history_data = {
            "epochs": np.array(self.epoch_checkpoints),
            "loss_history": np.array(self.loss_checkpoints),
        }

        np.savez_compressed(filename, **history_data)
        logger.info(f"Complete training history saved to {filename}")


class LoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_frequency=100, ndata=None):
        super().__init__()
        self.log_frequency = log_frequency
        self.n_data = ndata
        if ndata is None:
            self.n_data = 1

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_frequency == 0:
            loss = logs.get("loss", 0)
            total_str = f"Epoch {epoch+1}: Loss: {loss}, Loss/Ndat: {loss/self.n_data}"
            logger.info(total_str)
