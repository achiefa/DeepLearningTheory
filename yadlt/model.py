"""
PDFmodel class

Inspiration from n3fit: https://github.com/NNPDF/nnpdf/tree/master/n3fit
"""

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers as Kinit
import yaml

from yadlt.layers import InputScaling, Preprocessing

h5py_logger = logging.getLogger("h5py")
h5py_logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
tf.get_logger().setLevel("ERROR")

MAX_LAYER = 100000


def helper_zero_kernel_initializer(**args):
    """
    Wrapper for a zero kernel initializer.
    """
    return Kinit.Zeros


supported_kernel_initializers = {
    "RandomNormal": (Kinit.RandomNormal, {"mean": 0.0, "stddev": 1, "seed": 0}),
    "HeNormal": (Kinit.HeNormal, {"seed": 0}),
    "GlorotNormal": (Kinit.GlorotNormal, {"seed": 0}),
    "zeros": (helper_zero_kernel_initializer, {}),
}

# supported_optmizer = {
#     "SGD": tf.optimizers.SGD,
#     "Adam": tf.optimizers.Adam,
# }


@tf.function(reduce_retracing=True)
def compute_ntk_static(inputs, model, outputs):
    """
    Optimized Neural Tangent Kernel computation.
    This function computes the NTK in a more efficient way by
    leveraging static shapes and avoiding unnecessary operations.

    This function avoids reretracing by using `tf.function` with `reduce_retracing=True`.

    Albeit a similar function exists in the model class, this one is optimized and
    avoid retracing by using static shapes and a single operation to compute the NTK.

    TODO: The two functions should be merged, but this requires some refactoring
    """
    with tf.GradientTape(persistent=False) as tape:
        predictions = model(inputs)
        jacobian = tape.jacobian(predictions, model.trainable_variables)

    # Get the actual batch size from predictions shape
    batch_size = tf.shape(predictions)[0]
    seq_length = tf.shape(predictions)[1]

    if outputs == 1:
        # Concatenate jacobians along the parameter dimension
        jac_list = []
        for jac in jacobian:
            # Reshape to (batch_size * seq_length, -1) to handle the sequence dimension
            jac_flat = tf.reshape(jac, (batch_size * seq_length, -1))
            jac_list.append(jac_flat)

        # Concatenate all parameter jacobians
        full_jacobian = tf.concat(jac_list, axis=1)  # Shape: [input_size, total_params]

        # Compute NTK in one operation
        ntk = tf.matmul(full_jacobian, full_jacobian, transpose_b=True)

    else:
        # Multiple outputs case
        raise NotImplementedError(
            "Multiple outputs optimization not implemented. Use original method."
        )

    return ntk


class Chi2:
    def __init__(self, covmat):
        """
        Initialize the Chi2 class with a covariance matrix.

        Parameters
        ----------
        covmat : pd.DataFrame
            Covariance matrix for the chi-squared computation.
        """
        self.covmat = covmat
        self._invcovmat = tf.constant(np.linalg.inv(covmat), dtype=tf.float32)

    def __call__(self, ytrue, ypred):
        res = ytrue - ypred
        return tf.einsum("bi,ij,bj -> b", res, self._invcovmat, res)


def generate_pdf_model(
    outputs=1,
    architecture=[25, 20],
    activations=["tanh", "tanh"],
    kernel_initializer="GlorotNormal",
    bias_initializer="zeros",
    user_ki_args: Dict = None,
    scaled_input=False,
    preprocessing=False,
    seed=0,
):

    # Load kernel initializer function and arguments
    try:
        ki_tuple = supported_kernel_initializers[kernel_initializer]
    except KeyError as e:
        raise NotImplementedError(
            f"[PDFmodel] kernel initializer not implemented: {kernel_initializer}"
        ) from e

    ki_function = ki_tuple[0]
    ki_args = ki_tuple[1]
    current_seed = seed

    # Modify initialization arguments if needed
    if user_ki_args is not None:
        for key, value in user_ki_args.items():
            if key in ki_args.keys() and value is not None:
                ki_args[key] = value

    input_layer = tf.keras.layers.Input(shape=(None, 1), name="xgrid")

    pdf_raw = tf.keras.Sequential(name="pdf_raw")

    if scaled_input:
        scaled_input = InputScaling()
        pdf_raw.add(scaled_input)

    for l_idx, layer in enumerate(architecture):
        ki_args_layer = ki_args.copy()
        ki_args_layer["seed"] = current_seed
        current_seed += 1  # Increment seed for next layer
        pdf_raw.add(
            tf.keras.layers.Dense(
                layer,
                activation=activations[l_idx],
                kernel_initializer=ki_function(**ki_args_layer),
                bias_initializer=bias_initializer,
                dtype=tf.float32,
                name="deep_layer_" + str(l_idx),
            )
        )

    # Update seed for output layer
    ki_args_output = ki_args.copy()
    ki_args_output["seed"] = current_seed
    pdf_raw.add(
        tf.keras.layers.Dense(
            outputs,
            activation="linear",
            kernel_initializer=ki_function(**ki_args_output),
            dtype=tf.float32,
            bias_initializer=bias_initializer,
            name="output_layer",
        )
    )

    if preprocessing:
        mm_layer = tf.keras.layers.Multiply()
        preprocessing_factor = Preprocessing()
        final_result = mm_layer(
            [pdf_raw(input_layer), preprocessing_factor(input_layer)]
        )
        return tf.keras.models.Model(input_layer, final_result, name="pdf")

    return tf.keras.models.Model(input_layer, pdf_raw(input_layer), name="pdf")


def load_trained_model(replica_dir, epoch=-1):
    """
    Load a trained model from the specified directory.

    Parameters
    ----------
    replica_dir : str
        Directory where the model is stored.
    epoch : int, optional
        Specific epoch to load. If None, loads the latest model.

    Returns
    -------
    tf.keras.Model
        The loaded Keras model.
    """
    replica_dir = Path(replica_dir)
    metadata_file = replica_dir.parent / "metadata.yaml"

    with open(metadata_file, "r") as f:
        metadata = yaml.safe_load(f)

    # Extract model configuration
    model_config = metadata["model_info"]
    args_config = metadata["arguments"]

    # For backward compatibility, ensure keys exist
    model_config.setdefault("use_scaled_input", False)
    model_config.setdefault("use_preprocessing", False)

    # Reconstruct the PDF model with same configuration
    pdf_model = generate_pdf_model(
        outputs=model_config["outputs"],
        architecture=model_config["architecture"],
        activations=[
            model_config["activations"]
            for _ in range(len(model_config["architecture"]))
        ],
        kernel_initializer="zeros",  # model_config["kernel_initializer"],
        bias_initializer="zeros",
        user_ki_args=None,
        seed=args_config["seed"],  # Use same seed as training
        scaled_input=model_config["use_scaled_input"],
        preprocessing=model_config["use_preprocessing"],
    )

    # Find and load weights
    weight_file = load_weights(replica_dir, epoch)

    # Load weights directly into PDF model
    try:
        pdf_model.load_weights(weight_file)
    except ValueError as e:  # Handle legacy
        model = pdf_model.layers[1]
        model.load_weights(weight_file)

    return pdf_model, metadata


def load_weights(replica_dir, epoch=-1):
    """
    Load weights for the PDF model from the specified directory.

    Parameters
    ----------
    replica_dir : str
        Directory where the model weights are stored.
    epoch : int, optional
        Specific epoch to load. If None, loads the latest model.

    Returns
    -------
    tf.keras.Model
        The PDF model with loaded weights.
    """
    # Find and load weights
    if epoch == -1:
        # Find the latest epoch
        weight_files = list(replica_dir.glob("epoch_*.weights.h5"))
        if not weight_files:
            raise FileNotFoundError(f"No weight files found in {replica_dir}")

        epochs = [int(f.name.split(".")[0].split("_")[-1]) for f in weight_files]
        latest_epoch = max(epochs)
        weight_file = replica_dir / f"epoch_{latest_epoch}.weights.h5"
    else:
        weight_file = replica_dir / f"epoch_{epoch}.weights.h5"

    if not weight_file.exists():
        raise FileNotFoundError(f"PDF weight file not found: {weight_file}")

    return weight_file


def get_preactivation(model, layer_idx, input_data):
    """
    Extract pre-activation values for a specific layer given input data a
    a sequential model.

    Args:
        model (tf.keras.Model): The Keras model from which to extract pre-activations.
        layer_idx (int): Index of the layer (0-indexed)
        input_data: Input data to the model
    """
    keras_model = model

    # Check if the shape of input_data is correct
    if len(np.array(input_data).shape) != 3:
        input_data = tf.reshape(input_data, (1, -1, 1))

    if layer_idx == 0:
        # For first layer, pre-activations are the inputs
        return input_data

    # Create a function that outputs the pre-activation (input to the layer)
    get_preact_fn = keras.Function(
        [keras_model.inputs[0]], [keras_model.layers[layer_idx].input]
    )

    # Get pre-activations
    return get_preact_fn([input_data])[0]


def compute_preactivation_product(model, layer_idx, input_data):
    """
    Compute the product of pre-activations for a given layer and input data.

    Args:
        model (tf.keras.Model): The Keras model from which to extract pre-activations.
        layer_idx (int): Index of the layer (0-indexed)
        input_data: Input data to the model

    Returns:
        tf.Tensor: Tensor containing the product of pre-activations.

          φ_{i1,α1}^(l) * φ_{i2,α2}^(l)
    """
    preactivations = get_preactivation(model, layer_idx, input_data)

    phi_alpha1_i1 = preactivations[0, :, :]  # Shape: [data, neuron]
    phi_alpha2_i2 = preactivations[0, :, :]  # Shape: [data, neuron]

    # Compute outer product
    tensor_product = tf.einsum("ai,bj->aibj", phi_alpha1_i1, phi_alpha2_i2)

    return tensor_product


def compute_K_by_layer(model_ensemble: list, layer_idx: int, input_data):
    """
    Compute the correlation matrix K for each layer in the model ensemble.

    Args:
      model_ensemble (list): List of Keras models.
      layer_idx (int): Index of the layer (0-indexed).
      input_data: Input data to the model.

    Returns:
      tf.Tensor: Tensor containing the average value of the correlation matrix K
      for the specified layer over the ensemble of models.

        K_{i1 α1,i2 α2} = < φ_{i1,α1}^(l) * φ_{i2,α2}^(l) >
    """
    phi_a1a2_phi_i1i2 = compute_preactivation_product(
        model_ensemble[0].layers[1], layer_idx, input_data
    )
    for model in model_ensemble[1:]:
        phi_a1a2_phi_i1i2 += compute_preactivation_product(
            model.layers[1], layer_idx, input_data
        )
    return phi_a1a2_phi_i1i2 / len(model_ensemble)


def sample_from_mvg(mean, covmat, num_samples=1000, batch_size=1000):
    """Sample from a multivariate Gaussian distribution. If the covariance matrix
    is singular, it computes the spectrum of the covariance and samples in the
    subspace of non-zero directions (eigenvalues).

    This function allows to generate multiple batches of samples. This is useful to
    estimate the uncertainty of a Monte Carlo integration.

    Parameters
    ----------
    mean : np.ndarray
        Mean of the multivariate Gaussian distribution.
    covmat : np.ndarray
        Covariance matrix of the multivariate Gaussian distribution.
    num_samples : int, optional
        Total number of samples to generate. Default is 1000.
    batch_size : int, optional
        Number of samples to generate in each batch. Default is 1000.
    """
    # Check if the sample size is divisible by the batch size
    if num_samples % batch_size != 0:
        raise ValueError("num_samples must be divisible by batch_size")

    try:
        L = np.linalg.cholesky(covmat)  # covariance = L @ L.T

        for _ in range(num_samples // batch_size):
            samples = np.random.normal(size=(batch_size, covmat.shape[0]))
            samples = mean + samples @ L.T
            yield samples

    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(covmat)
        non_zero_mask = eigvals > 1e-8
        effective_rank = np.sum(non_zero_mask)
        if effective_rank == 0:
            raise ValueError("Covariance matrix is singular and has no effective rank.")
        eigvals = eigvals[non_zero_mask]
        eigvecs = eigvecs[:, non_zero_mask]

        for _ in range(num_samples // batch_size):
            z = np.random.normal(0, 1, (batch_size, effective_rank))
            z_scaled = z * np.sqrt(eigvals)  # Scale by sqrt of eigenvalues
            samples = mean + (eigvecs @ z_scaled.T).T  # Project back to full space
            yield samples


def mc_integrate_from_mvg(func, mean, covmat, num_samples=1000, batch_size=1000):
    """
    Perform Monte Carlo integration of a function over a multivariate Gaussian distribution.

    Parameters
    ----------
    func : callable
        Function to integrate. It should accept a 2D array of shape (num_samples, dim) as input.
    mean : np.ndarray
        Mean of the multivariate Gaussian distribution.
    covmat : np.ndarray
        Covariance matrix of the multivariate Gaussian distribution.
    num_samples : int, optional
        Total number of samples to generate. Default is 1000.
    batch_size : int, optional
        Number of samples to generate in each batch. Default is 1000.

    Returns
    -------
    float
        Estimated value of the integral and its statistical uncertainty.
    """
    batch_results = []
    batch_generator = sample_from_mvg(mean, covmat, num_samples, batch_size)
    for batch in batch_generator:
        batch_results.append(np.mean([func(sample) for sample in batch], axis=0))

    res_mean = np.mean(batch_results, axis=0)
    res_std = np.std(batch_results)

    return res_mean, res_std
