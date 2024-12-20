import tensorflow as tf
from tensorflow.keras import ops
from typing import Dict

class Observable_layer(tf.keras.layers.Layer):
    def __init__(self, fk_dict: Dict , **kwargs):
        """
        Custom layer that contracts the output of the neural network with the FK tables
        to produce the theoretical prediction.

        Parametsrs
        ----------
        fk_dict: Dict
          Dictionary containing the FK tables for each experiment.
          Each FK table is a Tensor of shape (Ndat, x_grid, Flavours) 
          that will be used for contraction.
          It is assumed that all FK tables are padded such that
          x_grid = 50.
        """
        super(Observable_layer, self).__init__(**kwargs)
        self.FK = fk_dict

    def call(self, f_pred):
        """
        Performs the contraction between the output of the neural
        network (shape: (x_grid, flavours)) and the FK tables.

        Parameters
        ----------
        f_pred: Output of the neural network (shape: (x_grid, flavours)).

        Return
        -------
        Returns a dictionary {exp : predictions}, where predictions is an
        array of the Ndat predictions for a given experiment. 
        """

        # Perform the contraction: sum over the second and third axes
        g = {}
        for exp, fk  in self.A.items():
          g[exp] = tf.einsum('Iia, ia -> I ', fk, f_pred)
        return g

    # TODO
    # This might be necessary for serialization.
    # Maybe deprecated ...?
    def get_config(self):
      config = super().get_config()
      config.update({
        'FK': self.FK 
      })
      return config
    

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