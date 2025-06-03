"""
PDFmodel class

Inspiration from n3fit: https://github.com/NNPDF/nnpdf/tree/master/n3fit
"""

from functools import partial
import json
import logging
from typing import Callable, Dict

import keras.initializers as Kinit
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from yadlt.layers import MyDense

h5py_logger = logging.getLogger("h5py")
h5py_logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


supported_kernel_initializers = {
    "RandomNormal": (Kinit.RandomNormal, {"mean": 0.0, "stddev": 1, "seed": 0}),
    "HeNormal": (Kinit.HeNormal, {"seed": 0}),
    "GlorotNormal": (Kinit.GlorotNormal, {"seed": 0}),
}

supported_optmizer = {
    "SGD": tf.optimizers.SGD,
    "Adam": tf.optimizers.Adam,
}


def mse_loss(Cinv, y_true, y_pred, dtype="float32"):
    loss = 0
    for exp, pred in y_pred.items():
        Cinv_exp = tf.convert_to_tensor(
            Cinv.xs(level="dataset", key=exp).T.xs(level="dataset", key=exp).to_numpy(),
            name=f"Cinv_{exp}",
            dtype=dtype,
        )
        R = tf.convert_to_tensor(pred - y_true[exp], name=f"residue_{exp}", dtype=dtype)
        Cinv_R = tf.linalg.matvec(Cinv_exp, R)
        loss += 0.5 * tf.reduce_sum(tf.multiply(R, Cinv_R))
    return loss


def generate_loss(func, *args, **kwargs):
    return partial(func, *args, **kwargs)


def generate_mse_loss(Cinv):
    return generate_loss(mse_loss, Cinv)


@tf.function(reduce_retracing=True)
def _compute_ntk_static(inputs, model, outputs):
    """
    Optimized Neural Tangent Kernel computation.
    This function computes the NTK in a more efficient way by
    leveraging static shapes and avoiding unnecessary operations.

    This function avoids reretracing by using `tf.function` with `reduce_retracing=True`.
    """
    with tf.GradientTape(persistent=False) as tape:
        predictions = model(inputs)
        jacobian = tape.jacobian(predictions, model.trainable_variables)

    input_size = inputs.shape[0]

    if outputs == 1:
        # Concatenate jacobians along the parameter dimension
        jac_list = []
        for jac in jacobian:
            # Flatten dimension
            jac_flat = tf.reshape(jac, (input_size, -1))
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


class PDFmodel:
    """
    The PDFmodel

    Create a Neural Network with given parameters.

    Parameters
    ----------
    input: list
      Array of inputs. In the NNPDF case, the input is the x-grid. For the sake
      of this work, all FK tables have been padded with the same size of x-grid,
      which is 50.
    outputs: int
      Number of output nodes, i.e. the number of active flavours.
    architecture: list(int)
      Number of nodes in each layer of the Neural Network.
    activations: list(str)
      Activation function for each layer.
    kernel_initializer: str
      Name of the function used to initialize the weights in each layer.
      Default is 'GlorotNormal'.
    user_ki_args: dict
      Dictionary with the parameters needed to the kernel initializer
      function. The seed, which is one the of the possible parameters
      in the kernel.initializers, should not be inserted here. Default
      is None as `GlorotNormal` does not need any parameters, except for
      the seed which enters as an another parameter in this class initializer.
    seed: int
      Seed used in the random generation of weights. Each layer adds +1 to the
      previous layer's seed, starting from layer 0 with this seed. In this way,
      it always possible to reproduce the results.
    dtype: 'float32'
      Float precision to be used in the model.

    """

    def __init__(
        self,
        input: list,
        outputs: int,
        architecture=[25, 20],
        activations=["tanh", "tanh"],
        kernel_initializer="GlorotNormal",
        user_ki_args: Dict = None,
        seed=0,
        dtype="float32",
        dense_layer="MyLayer",
        init_bias=False,
    ):

        logger.info("Initializing PDFmodel...")
        # Check that the length of architecture and activations
        # is the same
        if len(architecture) != len(activations):
            raise Exception(
                "The length of activations must match the number of layers."
            )

        # Convert input in tf.tensor
        x = tf.convert_to_tensor(input)
        x = tf.reshape(x, shape=(-1, 1))
        self.inputs = x
        self.float_type = dtype
        self.seed = seed
        self.architecture = architecture
        self.activations = activations
        self.outputs = outputs
        self.init_bias = init_bias

        if dense_layer == "MyLayer":
            self.dense_layer = MyDense
        else:
            self.dense_layer = tf.keras.layers.Dense
        logger.debug(f"Using {self.dense_layer} as dense layer.")

        # Check if kernel initializer is supported
        try:
            ki_tuple = supported_kernel_initializers[kernel_initializer]
        except KeyError as e:
            raise NotImplementedError(
                f"[PDFmodel] kernel initializer not implemented: {kernel_initializer}"
            ) from e

        # Store kernel_initializer function and arguments
        ki_function = ki_tuple[0]
        ki_args = ki_tuple[1]
        ki_args["seed"] = self.seed

        # Override defaults with user provided values
        if user_ki_args is not None:
            for key, value in user_ki_args.items():
                if key in ki_args.keys() and value is not None:
                    ki_args[key] = value
        logger.debug(f"Using {kernel_initializer} as kernel initializer.")
        logger.debug(f"Initialization parameters: {ki_args}")

        self.model = self._generate_model(ki_function, ki_args)
        self.config = {
            "dense_layer": "Dense",
            "input": str(input),
            "architecture": self.architecture,
            "activations": self.activations,
            "outputs": self.outputs,
            "seed": self.seed,
            "dtype": self.float_type,
        }

    def _generate_model(self, ki_function, ki_args):
        """
        Generate the Sequential model given the kernel initializer
        function and its arguments. This function is meant to be used
        internally and it is not part of the API.
        """
        logger.info("Generating model...")
        # Initialize keras sequential model
        model = keras.models.Sequential()

        # Select bias initializer
        if self.init_bias:
            bias_initializer = Kinit.RandomNormal(mean=0.0, stddev=1, seed=self.seed)
        else:
            bias_initializer = Kinit.Zeros()

        # Add input layer
        model.add(
            keras.Input(shape=self.inputs.shape, dtype=self.float_type, name="xgrid")
        )
        # np.random.seed(ki_args['seed'])
        # ki_args['seed'] = np.random.randint()

        # Loop over architecture and add layers
        for l_idx, layer in enumerate(self.architecture):
            model.add(
                self.dense_layer(
                    layer,
                    activation=self.activations[l_idx],
                    kernel_initializer=ki_function(**ki_args),
                    dtype=self.float_type,
                    bias_initializer=bias_initializer,
                    name="deep_layer_" + str(l_idx),
                )
            )

        # Add last layer
        ki_args["seed"] = np.random.randint(l_idx + 2)
        model.add(
            self.dense_layer(
                self.outputs,
                activation="linear",
                kernel_initializer=ki_function(**ki_args),
                dtype=self.float_type,
                bias_initializer=bias_initializer,
                name="output_layer",
            )
        )

        return model

    def call(self, inputs):
        """
        Custom implementation of the model's forward pass in call().
        """
        X = inputs
        if not isinstance(X, tf.Tensor):
            X = tf.convert_to_tensor(inputs)
            X = tf.reshape(X, shape=(-1, 1))
        return self.model(X)

    def __call__(self, inputs):
        return self.call(inputs)

    def predict(self, squeeze=False):
        """
        Similar to call, give PDF predictions using the attribute input.
        """
        if squeeze:
            return tf.squeeze(self.call(self.inputs))
        else:
            return self.call(self.inputs)

    def compute_predictions(self, FK_dict: Dict):
        """
        Compute the data predictions using the model. The outputs of
        the model are contracted with the FK tables.

        Parameters
        ----------
        FK_dict: dict
          Dictionary of FK tables for each experiment.

        Returns
        -------
        Dictionary where the key is the name of the experiment, and
        the respective value is the theoretical prediction computed
        using this model.
        """
        pdf_preds = tf.squeeze(self.predict())
        g = {}
        for exp, fk in FK_dict.items():
            if len(fk.shape) == 3:
                g[exp] = tf.einsum("Iia, ia -> I ", fk, pdf_preds)
            else:
                g[exp] = tf.einsum("Ii, i -> I ", fk, pdf_preds)
        return g

    def train_network_gd(
        self,
        data: Dict,
        FK_dict: Dict,
        loss_func: Callable,
        learning_rate: float,
        tol=1.0e-8,
        logging=False,
        callback=True,
        log_fr=100,
        max_epochs: int = 1e5,
        savedir=None,
        optimizer="SGD",
    ):
        """
        Train the model using SGD. If allowed, this method stores relevant
        objects (e.g. ntk) at each epoch of the training process.

        Parameters
        ----------
        data: dict
          Dictionary of experimental central values. They key is the name of the
          experiment, the value is the array of data.
        FK_dict: dict
          Dictionary of fk_tables for each experiment. They key is the name of the
          experiment, the value is the relative FK table.
        loss_func: func
          Function that takes y_pred and y_true as dictionaries and returns the
          the value of the loss.

        """
        optimizer = supported_optmizer[optimizer](learning_rate=learning_rate)
        logger.info(f"Using {optimizer} as optimizer.")

        ndata = tf.experimental.numpy.sum([i.size for i in data.values()])
        epoch = 0

        # TODO callback function handler should be implemented
        if callback and savedir is not None:
            logger.debug(f"Saving model config in {savedir}")
            config = self.config
            with open(savedir / "config.json", "w") as f:
                json.dump(config, f)

            # TODO only BCDMS data are saved
            np.save(savedir / "data", data["BCDMS"])

            logger.debug(f"Weights of the model saved in {savedir}")
            self.model.save_weights(savedir / f"epoch_{epoch}.weights.h5")

        try:
            while True:
                with tf.GradientTape(persistent=False) as tape:
                    # Forward pass: Compute predictions
                    preds = self.compute_predictions(FK_dict)

                    # Compute loss
                    # f = self.predict()
                    # loss =  tf.add(loss_func(y_true=data, y_pred=preds, dtype=self.float_type), tf.multiply(tf.multiply(tf.norm(f), tf.norm(f)), 0.5**2))
                    loss = loss_func(y_true=data, y_pred=preds, dtype=self.float_type)

                    if epoch == 0:
                        loss_pre = loss
                        rel_loss = 0.0
                    else:
                        rel_loss = abs(loss - loss_pre) / loss_pre
                        if rel_loss > tol and epoch != 0:
                            loss_pre = loss
                        elif rel_loss < tol:
                            if callback and savedir is not None:
                                self.model.save_weights(
                                    savedir / f"epoch_{epoch}.weights.h5"
                                )
                            logger.warning(f"Convergence reached at epoch {epoch}.")
                            break

                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    optimizer.apply_gradients(
                        zip(gradients, self.model.trainable_variables)
                    )

                if logging:
                    if epoch % log_fr == 0:
                        total_str = f"Epoch {epoch}/{max_epochs}:\n   Loss: {loss.numpy()}, Loss/Ndat: {loss.numpy()/ndata}, Rel. loss: {rel_loss}"
                        logger.info(total_str)
                epoch += 1

                if epoch > max_epochs:
                    print("Maximum number of iterations reached.")
                    if callback and savedir is not None:
                        self.model.save_weights(savedir / f"epoch_{epoch}.weights.h5")
                    break

                # Save model and epochs
                if epoch % log_fr == 0 and callback and savedir is not None:
                    self.model.save_weights(savedir / f"epoch_{epoch}.weights.h5")
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user with Ctrl+C.")
            if callback and savedir is not None:
                self.model.save_weights(savedir / f"epoch_{epoch}.weights.h5")
                logger.info(f"Model saved in {savedir} at epoch {epoch}.")

    def compute_ntk_optimized(self):
        """Wrapper method that calls the static function."""
        return _compute_ntk_static(self.inputs, self.model, self.outputs)

    def compute_ntk(self, only_diagonal=False, learning_tensor=False):
        """
        Compute the Neural Tanget Kernel (NTK) of the Sequential model.

        The NTk is compute as the contraction of the derivatives of the
        outputs with respect to each parameter of the network.

        """
        with tf.GradientTape(persistent=False) as tape:
            predictions = self.model(self.inputs)
            jacobian = tape.jacobian(predictions, self.model.trainable_variables)

        # Initialize the ntk
        input_size = self.inputs.shape[0]

        if self.outputs == 1:
            ntk = tf.zeros((input_size, input_size), dtype=self.float_type)
        else:
            ntk = tf.zeros(
                (input_size, self.outputs, input_size, self.outputs),
                dtype=self.float_type,
            )
            raise Warning(
                "TODO: The implementation for multiple outputs needs to be checked."
            )

        for jac in jacobian:
            jac = tf.squeeze(jac, axis=[1, 2])
            if len(jac.shape) == 3:
                jac_contracted = tf.tensordot(jac, jac, axes=([-1, -2], [-1, -2]))
                if learning_tensor:
                    jac_contracted /= jac.shape[1]
            elif len(jac.shape) == 2:
                jac_contracted = tf.tensordot(jac, jac, axes=([-1], [-1]))
            ntk += jac_contracted

        try:
            transposed_ntk = tf.transpose(
                ntk, perm=[2, 3, 0, 1] if len(ntk.shape) == 4 else [1, 0]
            )
            assert tf.experimental.numpy.allclose(ntk, transposed_ntk)
        except AssertionError:
            raise RuntimeError("The NTK is not symmetric. Check the implementation.")

        if only_diagonal:
            ntk_diaogonal = tf.Variable(
                tf.zeros(
                    (input_size, self.outputs, input_size, self.outputs),
                    dtype=self.float_type,
                )
            )
            for f in range(self.outputs):
                ntk_diaogonal[:, f, :, f].assign(ntk[:, f, :, f])
            ntk = ntk_diaogonal

        return ntk

    @classmethod
    def load_model(self, config_path, weights_path):
        """
        Load the model from a json file and the weights from a h5 file.
        """
        with open(config_path, "r") as f:
            config = json.load(f)

        # Create the model
        config["input"] = np.fromstring(config["input"].strip("[]"), sep=" ")
        model = PDFmodel(**config)

        # Load the weights
        model.model.load_weights(weights_path)

        return model
