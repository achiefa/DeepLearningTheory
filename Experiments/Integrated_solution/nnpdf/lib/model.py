"""
    PDFmodel class

    Inspiration from n3fit: https://github.com/NNPDF/nnpdf/tree/master/n3fit
"""

import tensorflow as tf
import tensorflow.keras as keras
import keras.initializers as Kinit
from functools import partial
import copy

from typing import Dict, Callable

from layers import MyDense, Observable_layer

supported_kernel_initializers = {
  'RandomNormal': (Kinit.RandomNormal, {'mean': 0.0, 'stddev': 1, 'seed': 0}),
  'HeNormal': (Kinit.HeNormal, {'seed': 0}),
  'GlorotNormal': (Kinit.GlorotNormal, {'seed': 0}),
}

def mse_loss(Cinv, y_true, y_pred, dtype='float32'):
  loss = 0
  for exp, pred in y_pred.items():
      Cinv_exp = tf.convert_to_tensor(Cinv.xs(level="dataset", key=exp).T.xs(level="dataset", key=exp).to_numpy(), name=f'Cinv_{exp}', dtype=dtype)
      R = tf.convert_to_tensor(pred - y_true[exp], name=f'residue_{exp}', dtype=dtype)
      Cinv_R = tf.linalg.matvec(Cinv_exp, R)
      loss += 0.5 * tf.reduce_sum(tf.multiply(R, Cinv_R))
  return loss

def generate_loss(func, *args, **kwargs):
   return partial(func, *args, **kwargs)

def generate_mse_loss(Cinv):
  return generate_loss(mse_loss, Cinv)

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
  def __init__(self,
               input: list,
               outputs: int,
               architecture=[25,20],
               activations=['tanh', 'tanh'],
               kernel_initializer='GlorotNormal',
               user_ki_args: Dict=None,
               seed=0,
               dtype='float32'):
    # Check that the length of architecture and activations
    # is the same
    if len(architecture) != len(activations):
      raise Exception("The length of activations must match the number of layers.")
    
    # Convert input in tf.tensor
    x = tf.convert_to_tensor(input)
    x = tf.reshape(x, shape=(-1,1))
    self.inputs = x
    self.float_type = dtype
    self.seed = seed
    self.architecture = architecture
    self.activations = activations
    self.outputs = outputs

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
    ki_args['seed'] = self.seed

    # Override defaults with user provided values
    if user_ki_args is not None:
      for key, value in user_ki_args.items():
        if key in ki_args.keys() and value is not None:
          ki_args[key] = value

    self.model = self.__generate_model(ki_function, ki_args)


  def __generate_model(self, ki_function, ki_args):
    """
    Generate the Sequential model given the kernel initializer
    function and its arguments. This function is meant to be used
    internally and it is not part of the API.
    """
    # Initialize keras sequential model
    model = keras.models.Sequential()

    # Add input layer
    model.add(keras.Input(shape=self.inputs.shape, dtype=self.float_type, name='xgrid'))

    # Loop over architecture and add layers
    for l_idx, layer in enumerate(self.architecture):
      ki_args['seed'] += l_idx
      model.add(MyDense(layer,
                        activation=self.activations[l_idx],
                        kernel_initializer=ki_function(**ki_args),
                        dtype=self.float_type))
    
    # Add last layer
    ki_args['seed'] += 1
    model.add(MyDense(self.outputs,
                      activation='linear',
                      kernel_initializer=ki_function(**ki_args),
                      dtype=self.float_type))
    
    return model
  
  def call(self, inputs):
    """
    Custom implementation of the model's forward pass in call().
    """
    X = inputs
    if not isinstance(X, tf.Tensor):
      X = tf.convert_to_tensor(inputs)
      X = tf.reshape(X, shape=(-1,1))
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
    for exp, fk  in FK_dict.items():
      g[exp] = tf.einsum('Iia, ia -> I ', fk, pdf_preds)
    return g

  def train_network_gd(self,
                       data: Dict,
                       FK_dict: Dict,
                       loss_func: Callable,
                       learning_rate: float,
                       tol=1.e-8,
                       logging=False,
                       callback=True,
                       cb_time_range=100):
    """
    Train the model using SGD. If allowed, this method stores relevant
    objects (e.g. ntk) at each step of the training process.

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
    X = self.inputs
    optimizer = tf.optimizers.SGD(learning_rate=learning_rate)
    ndata= tf.experimental.numpy.sum([i.size for i in data.values()])
    step = 0

    if callback:
      model_in_time = []
      saved_steps = []

    while True:
      with tf.GradientTape(persistent=False) as tape:
        # Forward pass: Compute predictions
        preds = self.compute_predictions(FK_dict)

        # Compute loss
        loss = loss_func(y_true=data, y_pred=preds, dtype=self.float_type)

        # Save model and steps
        if step % cb_time_range == 0:
          model_in_time.append(copy.deepcopy(self))
          saved_steps.append(step)
    
        if step == 0:
          loss_pre = loss
          rel_loss = 0.0
        else: 
          rel_loss = abs(loss - loss_pre)/loss_pre
          if rel_loss > tol and step != 0:
            loss_pre = loss
          elif rel_loss < tol:
            break

        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

      if logging:
        if step % 100 == 0:
          print('------------------------')
          print(f"Step {step}, Loss: {loss.numpy()}, Loss/Ndat: {loss.numpy()/ndata}, Rel. loss: {rel_loss}")
      step += 1
    
    if callback:
      return (model_in_time, saved_steps)

  def copmute_jacobian(self):
    """
    Compute the jacobian of the model.
    """
    with tf.GradientTape(persistent=False) as tape:
      #tape.watch(x)
      # Forward pass
      predictions = self.predict()
      jacobian = tape.jacobian(predictions, self.model.trainable_variables)
    return jacobian

  def compute_ntk(self, only_diagonal=False):
    """
    Compute the Neural Tanget Kernel (NTK) of the Sequential model.

    The NTk is compute as the contraction of the derivatives of the
    outputs with respect to each parameter of the network.

    """
    with tf.GradientTape(persistent=False) as tape:
      #tape.watch(x)
      # Forward pass
      predictions = self.model(self.inputs)
      jacobian = tape.jacobian(predictions, self.model.trainable_variables)

    # Initialize the ntk
    input_size = self.inputs.shape[0]
    ntk = tf.zeros((input_size, self.outputs, input_size, self.outputs), dtype=self.float_type)
    for jac in jacobian:
      jac = tf.squeeze(jac)
      dot_axes = [-1,-2] if len(jac.shape) == 4 else [-1]
      jac_contracted = tf.tensordot(jac, jac, axes=(dot_axes, dot_axes))
      ntk += jac_contracted

    try:
      transposed_ntk = tf.transpose(ntk, perm=[2, 3, 0, 1])
      assert(tf.experimental.numpy.allclose(ntk, transposed_ntk))
    except AssertionError:
      raise RuntimeError('The NTK is not symmetric. Check the implementation.')
    
    if only_diagonal:
      ntk_diaogonal = tf.Variable(tf.zeros((input_size, self.outputs, input_size, self.outputs), dtype=self.float_type))
      for f in range(self.outputs):
        ntk_diaogonal[:,f,:,f].assign(ntk[:,f,:,f])
      ntk = ntk_diaogonal
    
    return ntk