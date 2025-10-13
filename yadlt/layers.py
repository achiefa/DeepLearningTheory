import numpy as np
import tensorflow as tf
from tensorflow.keras import ops


class Preprocessing(tf.keras.layers.Layer):
    """This layer represents the preprocessing function that multiplies
    the model by (1 - x)**(1 + beta)"""

    def build(self, input_shape):
        self._beta = self.add_weight(
            shape=(1,),
            trainable=True,
            name="beta",
            constraint=tf.keras.constraints.non_neg(),
            initializer="ones",
        )

    def call(self, x):
        return (1.0 - x) ** (self._beta + 1.0)


class InputScaling(tf.keras.layers.Layer):
    """This layer applies the logarithmic scaling to the input
    and concatenates it.
    """

    def call(self, x):
        return tf.concat([x, tf.math.log(x) / tf.math.log(10.0)], axis=-1)


class Convolution(tf.keras.layers.Layer):
    """Applies the convolution operation to the output of the model"""

    def __init__(self, fktables, basis, nfl=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fktables = tf.constant(fktables, dtype=tf.float32)
        basis_mask = np.zeros(nfl, dtype=np.float32)
        for i in basis:
            basis_mask[i] = True
        self._basis_mask = tf.constant(basis_mask)

    def call(self, pdf):
        """Convolution operation"""
        pdf_reshaped = tf.squeeze(pdf, axis=0)  # Remove batch dimension
        masked_pdf = tf.boolean_mask(pdf_reshaped, self._basis_mask, axis=1)
        return tf.einsum("nfx, xf -> n", self._fktables, masked_pdf)


class MyDense(tf.keras.layers.Dense):
    """
    Wrapper around the Dense layer defined in keras. This redefinition
    of the Dense layer enforces the NTK parametrization of the pre-activation
    :math:`$z_i^{(l)} = b_i^{(l)} + 1 / sqrt(N_{in}) W_{ij} \rho_j^{(l-1)}$`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        """
        Redefinition of the `call` method such that the
        kernel is dived by the size of the previous layer.
        """
        x = ops.matmul(inputs, self.kernel)
        x = ops.divide(x, ops.sqrt(self.kernel.shape[0]))
        if self.bias is not None:
            x = ops.add(x, self.bias)
        if self.activation is not None:
            x = self.activation(x)
        return x
