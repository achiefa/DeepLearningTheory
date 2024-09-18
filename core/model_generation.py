import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

def generate_sequential_model(outputs=1, 
                   input_layer=None, 
                   nlayers=2, 
                   units=[100,100],
                   initialisation="True",
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
  
  if kwargs.get("seed"):
    seed = kwargs.get("seed")
  else:
    seed = "seed"
  
  if kwargs.get("kernel_initialiser"):
    kernel_initialiser = kwargs.get("kernel_initialiser")
  else:
    kernel_initialiser = tf.keras.initializers.GlorotNormal

  if kwargs.get("bias_initialiser"):
    bias_initialiser = kwargs.get("bias_initialiser")
  else:
    bias_initialiser = "zeros"     

  if kwargs.get("act_func"):
    act_func = kwargs.get("act_func")
  else:
    act_func = "tanh"

  if kwargs.get("out_func"):
    out_func = kwargs.get("out_func")
  else:
    out_func = "linear"
  
  model = tf.keras.models.Sequential(name="pdf")
  if input_layer is not None:
      model.add(input_layer)
  for layer in range(nlayers):
      model.add(tf.keras.layers.Dense(units[layer], 
                                      activation=act_func,
                                      kernel_initializer="glorot_uniform",
                                      bias_initializer="zeros"),
      )
  model.add(tf.keras.layers.Dense(outputs, activation=out_func))

  if initialisation:
    opt = tf.keras.optimizers.Nadam()
    model.compile(opt, loss="mse")

  return model


def generate_models_replicas(number_of_replicas=100, 
                             architecture=[10,10,1],
                             act_func="tanh",
                             model_generator=generate_sequential_model,
                             initialisation=True,
                             seed=1000,
                             **kwargs):
  """
  Generate an ensemble of models generated with `model_generator`. Note that
  `model_generator` is defaulted to `generate_sequential_model`, which creates
  a sequential model (i.e. a neural network) given the architecture and the parameters.

  Arguments:
    number_of_replicas: int (default=100)
        The size of the ensemble, namely the number of models generated.
    architecture: int (default=[10,10,1])
        The architecture of the network starting from the first deep layer.
    act_func: str (default="tanh")
        The activation function in the deep layers of the model.
    model_generator: func -> model (default=generate_sequential_model)
        The function that generates the model.
    output: str (default="linear")
        The type of linear output of the model
    initialisation: bool (defaylt=True)
        Whether or not using the models only at initialisation.
    seed: int (default=1000)
        Seed for the model generation

  Return: list
    A list containing the models
  """
  nn_models = []
  for _ in range(number_of_replicas):
    nn_models.append(
      model_generator(outputs=architecture[-1], 
                      nlayers=len(architecture)-1, 
                      units=architecture[:-1], 
                      activation=act_func,
                      initialisation=initialisation,
                      seed=seed + _,
                      **kwargs),
    )
  
  return nn_models