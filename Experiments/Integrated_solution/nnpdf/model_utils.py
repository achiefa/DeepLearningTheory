import tensorflow as tf
from multiprocessing import Pool
import numpy as np
from functools import partial
from tensorflow.keras import ops

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

    def get_config(self):
      config = super().get_config()
      config.update({
        'A': self.A 
      })
      return config
    

class MyDense(tf.keras.layers.Dense):
  def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

  def call(self, inputs):
        x = ops.matmul(inputs, self.kernel)
        x = ops.divide(x, ops.sqrt(self.kernel.shape[0]))
        if self.bias is not None:
            x = ops.add(x, self.bias)
        if self.activation is not None:
            x = self.activation(x)
        return x



def generate_sequential_model(outputs=1, 
                   input_layer=None, 
                   nlayers=2, 
                   units=[100,100],
                   seed=0,
                   predictions=False,
                   dtype='float32',
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
  
  if kwargs.get('kernel_initializer') == 'RandomNormal':
      kernel_initializer = tf.keras.initializers.RandomNormal
  elif kwargs.get('kernel_initializer'):
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
      model.add(tf.keras.layers.InputLayer(shape=(50,),name='input',  dtype=dtype))
  for layer in range(nlayers):
      model.add(MyDense(units[layer], 
                                      activation=activation_list[layer],
                                      kernel_initializer=kernel_initializer(mean=0., stddev=1, seed=seed - layer),
                                      dtype=dtype),
      )
  model.add(MyDense(outputs, 
                                  activation=output_func, 
                                  kernel_initializer=kernel_initializer(mean=0., stddev=1, seed=seed - layer),
                                  dtype=dtype))
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
  

def compute_ntk(model, input, only_diagonal=False):
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
      # Somehow einsum wants float32
      jac = tf.cast(jac, dtype=tf.float32)
      jac_concat = tf.reshape(jac, (jac.shape[0], jac.shape[1], prod)) 
      ntk += tf.einsum('iap,jbp->iajb', jac_concat, jac_concat)

  if only_diagonal:
      ntk_diaogonal = np.zeros((batch_size, n_outputs, batch_size, n_outputs))
      for f in range(n_outputs):
        ntk_diaogonal[:,f,:,f] = ntk[:,f,:,f]    
      ntk = ntk_diaogonal
  
  # Deprecated
  # if round_to_zero:
  #     # Set the small eigenvalues to zero for round-off.
  #     # Then reconstruct the original matrix 

  #     # First, flatten the ntk into a matrix
  #     oldshape = np.array(ntk.shape)
  #     prod = 1
  #     for k in oldshape[2:]:
  #       prod *= k
  #     try:
  #        ntk = ntk.numpy()
  #        ntk_flatten = ntk.reshape(prod, -1)
  #     except AttributeError:
  #       ntk_flatten = ntk.reshape(prod, -1)

  #     # Compute eigenvalues and eigenfunction of the flatten NTK.
  #     # Then rouond to zero
  #     eigvals, eigvecs = np.linalg.eig(ntk_flatten)

  #     # Forget about the imaginary part
  #     eigvals = eigvals.real
  #     eigvecs = eigvecs.real
  #     for i in range(eigvals.size):
  #         eigvals[i] = round_float32(eigvals[i], np.sort(eigvals)[::-1][0])
  #     reconstructed = np.dot(eigvecs * eigvals, eigvecs.T)
  #     ntk = reconstructed.reshape(oldshape)
  
  return ntk




from validphys.theorycovariance.output import matrix_plot_labels
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_covmat_heatmap(covmat, title):
    """Matrix plot of a covariance matrix."""
    df = covmat

    matrix = df.values
    fig, ax = plt.subplots(figsize=(15, 15))
    matrixplot = ax.matshow(
        matrix,
        cmap=cm.Spectral_r,
        norm=mcolors.SymLogNorm(
            linthresh=0.00001, linscale=1, vmin=-matrix.max(), vmax=matrix.max()
        ),
    )

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)

    cbar = fig.colorbar(matrixplot, cax=cax)
    cbar.set_label(label=r"$\widetilde{P}$", fontsize=20)
    cbar.ax.tick_params(labelsize=20)
    ax.set_title(title, fontsize=25)
    ticklocs, ticklabels, startlocs = matrix_plot_labels(df)
    ax.set_xticks(ticklocs)
    ax.set_xticklabels(ticklabels, rotation=30, ha="right", fontsize=20)
    ax.xaxis.tick_bottom()
    ax.set_yticks(ticklocs)
    ax.set_yticklabels(ticklabels, fontsize=20)
    
    # Shift startlocs elements 0.5 to left so lines are between indexes
    startlocs_lines = [x - 0.5 for x in startlocs]
    ax.vlines(startlocs_lines, -0.5, len(matrix) - 0.5, linestyles="dashed")
    ax.hlines(startlocs_lines, -0.5, len(matrix) - 0.5, linestyles="dashed")
    ax.margins(x=0, y=0)
    return fig


def train_network(model, 
                 optimizer, 
                 input_model, 
                 Cinv, 
                 data, 
                 show_log=False, 
                 ntk_round_to_zero=False, 
                 ntk_only_diagonal=False, 
                 tol=1.e-8):
  X = input_model
  predictions_in_time = []
  pdfs_in_time = []
  NTK_in_time = []

  saved_steps = []
  step = 0
  loss_pre = 0
  while True:  # Number of epochs/iterations
    # Save data
    if step % 100 == 0:
      pdf_from_model = tf.keras.models.Sequential(model.layers[:-1])
      
      NTK = compute_ntk(pdf_from_model, X.numpy(), round_to_zero=ntk_round_to_zero, only_diagonal=ntk_only_diagonal)
      NTK = np.array(NTK)
      prod = 1
      oldshape = NTK.shape
      for k in oldshape[2:]:
          prod *= k
      NTK = NTK.reshape(prod,-1)

      pdfs_in_time.append(pdf_from_model(X))
      NTK_in_time.append(NTK)
      saved_steps.append(step)

    with tf.GradientTape(persistent=False) as tape:
        # Forward pass: Compute predictions
        predictions = model(X)
        #if step % 100 == 0:
        predictions_in_time.append(predictions)

        loss = 0
        for exp, pred in predictions.items():
          Cinv_exp = tf.convert_to_tensor(Cinv.xs(level="dataset", key=exp).T.xs(level="dataset", key=exp).to_numpy(), name=f'Cinv_{exp}', dtype='float32')
          R = tf.convert_to_tensor(pred - np.array(data[exp]), name=f'residue_{exp}', dtype='float32')
          Cinv_R = tf.linalg.matvec(Cinv_exp, R)
          loss += 0.5 * tf.reduce_sum(tf.multiply(R, Cinv_R))
        
        if step == 0:
          loss_pre = loss
        else: 
          if abs(loss - loss_pre)/loss_pre > tol and step != 0:
             loss_pre = loss
          elif abs(loss - loss_pre)/loss_pre < tol:
             break

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if show_log:
          if step % 100 == 0:
            print('------------------------')
            print(f"Step {step}, Loss: {loss.numpy()}")
    step += 1
  
  return (NTK_in_time, predictions_in_time, pdfs_in_time, saved_steps)

from concurrent.futures import ProcessPoolExecutor, as_completed