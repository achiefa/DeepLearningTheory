import tensorflow as tf
import numpy as np

class ComputeObservable(tf.keras.layers.Layer):
    def __init__(self, fk_dict, **kwargs):
        """
        Custom layer that contracts the output of the neural network with the FK tables.

        Parametsrs
        ----------
        A_tf: Dictionary containing the FK tables for each experiment.
              Each FK table is a Tensor of shape (Ndat, Xgrid, Flavours) 
              that will be used for contraction.
        """
        super(ComputeObservable, self).__init__(**kwargs)
        self.A = fk_dict

    def call(self, f_pred):
        """
        Performs the contraction between f_pred (shape: (xgrid, flavours)) 
        and the FK tables.

        Parameters
        ----------
        f_pred: Output of the neural network (shape: (Xgrid, flavours)).

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
    


def generate_sequential_model(outputs=1, 
                   input_layer=None, 
                   nlayers=2, 
                   units=[100,100],
                   seed=0,
                   predictions=False,
                   **kwargs):
  """
  Create a tensorflow sequential model where all intermediate layers have the same size
  This function accepts an already constructed layer as the input.

  All hidden layers will have the same number of nodes for simplicity

  Arguments:
      outputs: int (default=1)
          number of output nodes (how many flavours are we training)
      input_layer: KerasTensor (default=None)
          if given, sets the input layer of the sequential model
      nlayers: int
          number of hidden layers of the network
      units: int
          number of nodes of every hidden layer in the network
      activation: str
          activation function to be used by the hidden layers (ex: 'tanh', 'sigmoid', 'linear')
  """
  if len(units) != nlayers:
      raise Exception("The length of units must match the number of layers.")
  
  if kwargs.get('kernel_initializer'):
      kernel_initializer = kwargs['kernel_initializer']
  else:
      kernel_initializer = tf.keras.initializers.HeNormal

  if kwargs.get('activation_list'):
      activation_list = kwargs['activation_list']
      if len(units) != len(activation_list):
          raise Exception("The length of the activation list must match the number of layers.")
  else:
      activation_list = ['tanh' for _ in range(nlayers)]

  if kwargs.get('output_func'):
      output_func = kwargs['output_func']
  else:
      output_func = 'linear'
  
  if kwargs.get('name'):
      name = kwargs['name']
  else:
      name = 'pdf'
  
  model = tf.keras.models.Sequential(name=name)
  if input_layer is not None:
      model.add(tf.keras.layers.InputLayer(shape=(50,),name='input'))
  for layer in range(nlayers):
      model.add(tf.keras.layers.Dense(units[layer], 
                                      activation=activation_list[layer],
                                      kernel_initializer=kernel_initializer(seed=seed - layer),
                                      ),
      )
  model.add(tf.keras.layers.Dense(outputs, 
                                  activation=output_func, 
                                  kernel_initializer=kernel_initializer(seed=seed - nlayers)
                                  ))
  if predictions:  
    model.add(ComputeObservable(kwargs.get('fk_table_dict')))

  return model


def round_float32(value, ref_value, tol_magnitude=1.e-6):
  """
  Utility function used to round to zero according to a reference value.
  """
  ratio = abs(value) / abs(ref_value)
  if ratio < tol_magnitude:
    return 0.0
  else:
    return value
  

def compute_ntk(model, input, only_diagonal=True, round_to_zero=False):
  """
  Copmute the Neural Tanget Kernel (NTK) of a Sequential model constructed
  with the Keras API.

  Parameters
  ----------
  model: Keras Sequential model
  input: The batch input used to evaluate the model
  only_diagonal: Return only diagonal (in the network outputs) elements
    of the NTK.
  round_to_zero: Reconstruct the NTK from the spectral decomposition 
    by first rounding to zero its eigenvalues.
  """
  # Record operations for gradient computation
  batch_size = input.size
  n_outputs = model.layers[-1].units
  x = tf.convert_to_tensor(input)
  x = tf.reshape(x, shape=(-1,1))
  with tf.GradientTape(persistent=False) as tape:
      #tape.watch(x)
      # Forward pass
      predictions = model(x)

  jacobian = tape.jacobian(predictions, model.trainable_variables)  

  ntk = tf.zeros((batch_size, n_outputs, batch_size, n_outputs))
  for jac in jacobian:
      prod = np.prod(jac.shape[2:])
      jac_concat = tf.reshape(jac, (jac.shape[0], jac.shape[1], prod)) 
      ntk += tf.einsum('iap,jbp->iajb', jac_concat, jac_concat)

  if only_diagonal:
      ntk_diaogonal = np.zeros((batch_size, n_outputs, batch_size, n_outputs))
      for f in range(n_outputs):
        ntk_diaogonal[:,f,:,f] = ntk[:,f,:,f]    
      ntk = ntk_diaogonal
  
  if round_to_zero:
      # Set the small eigenvalues to zero for round-off.
      # Then reconstruct the original matrix 

      # First, flatten the ntk into a matrix
      oldshape = ntk.shape
      prod = 1
      for k in oldshape[2:]:
        prod *= k
      ntk_flatten = ntk.reshape(prod, -1)

      # Compute eigenvalues and eigenfunction of the flatten NTK.
      # Then rouond to zero
      eigvals, eigvecs = np.linalg.eig(ntk_flatten)

      # Forget about the imaginary part
      eigvals = eigvals.real
      eigvecs = eigvecs.real
      for i in range(eigvals.size):
          eigvals[i] = round_float32(eigvals[i], np.sort(eigvals)[::-1][0])
      reconstructed = np.dot(eigvecs * eigvals, eigvecs.T)
      ntk = reconstructed.reshape(oldshape)
  
  return ntk