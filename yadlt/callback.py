import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class WeightStorageCallback(tf.keras.callbacks.Callback):
    """Callback to store model weights every f epochs"""

    def __init__(self, storage_frequency=100, storage_path=None, training_data=None):
        super().__init__()
        self.storage_frequency = storage_frequency
        self.storage_path = storage_path or "model_weights"
        self.epoch_checkpoints = []
        self.loss_checkpoints = []
        self.training_data = training_data

        # Create storage directory
        Path(self.storage_path).mkdir(exist_ok=True)

    def on_train_begin(self, logs=None):
        """Save initial weights before training starts"""
        self.epoch_checkpoints.append(0)

        # Evaluate loss at initial state
        x_train, y_train = self.training_data
        predictions = self.model(x_train, training=True)
        initial_loss = self.model.compute_loss(
            x_train, y_train, predictions, training=True
        )
        self.loss_checkpoints.append(initial_loss)
        logger.info(f"Initial loss: {initial_loss}")

        # Get PDF model weights
        pdf_model = self.model.get_layer("Observable").get_layer("pdf")

        # Save weights to file
        weight_filename = f"{self.storage_path}/epoch_0.weights.h5"
        pdf_model.save_weights(weight_filename)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.storage_frequency == 0:
            # Store weights in memory
            self.epoch_checkpoints.append(epoch + 1)
            self.loss_checkpoints.append(logs.get("loss", 0))

            # Get PDF model weights
            pdf_model = self.model.get_layer("Observable").get_layer("pdf")

            # Save weights to file
            weight_filename = f"{self.storage_path}/epoch_{epoch + 1}.weights.h5"
            pdf_model.save_weights(weight_filename)

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


class NaNCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        if np.isnan(logs.get("loss")):
            logger.error(f"NaN detected at batch {batch}")
            self.model.stop_training = True


class GradientNormCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_frequency=100, clip_norm=None, training_data=None):
        super().__init__()
        self.log_frequency = log_frequency
        self.clip_norm = clip_norm
        self.training_data = training_data

        # Storage for gradient norms
        self.grad_norms = []
        self.epochs = []
        self.clipping_events = 0
        self.total_batches = 0

    def on_epoch_end(self, epoch, logs=None):
        x_train, y_train = self.training_data

        # Compute gradient norm
        with tf.GradientTape() as tape:
            predictions = self.model(x_train, training=True)
            loss_value = self.model.compute_loss(x_train, y_train, predictions)

        gradients = tape.gradient(loss_value, self.model.trainable_variables)
        grad_norm = tf.linalg.global_norm(gradients)

        # Store the gradient norm
        self.grad_norms.append(float(grad_norm))
        self.epochs.append(epoch)
        self.total_batches += 1

        # Check if clipping would occur
        if self.clip_norm and grad_norm > self.clip_norm:
            self.clipping_events += 1

        # Log periodically
        if epoch % self.log_frequency == 0:
            clipping_freq = (
                self.clipping_events / self.total_batches
                if self.total_batches > 0
                else 0
            )

            recent_norms = self.grad_norms[
                -min(self.log_frequency, len(self.grad_norms)) :
            ]
            stats = {
                "current_grad_norm": float(grad_norm),
                "mean_grad_norm": np.mean(recent_norms),
                "max_grad_norm": np.max(recent_norms),
                "min_grad_norm": np.min(recent_norms),
                "std_grad_norm": np.std(recent_norms),
                "clipping_frequency": clipping_freq,
                "total_clipping_events": self.clipping_events,
            }

            logging.info(
                f"Epoch {epoch} - Gradient norm: {grad_norm:.4f}, "
                f"Mean (last {len(recent_norms)}): {stats['mean_grad_norm']:.4f}, "
                f"Clipping freq: {clipping_freq:.2%}"
            )
