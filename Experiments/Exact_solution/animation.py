import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from IPython import display 
import scipy

def compute_ntk(model, input):
    grad = []
    for x in tf.convert_to_tensor(input):
        with tf.GradientTape() as tape:
            x = tf.reshape(x, shape=(-1,1))
            #tape.watch(x)
            pred = model(x)

        # compute gradients df(x)/dtheta
        g = tape.gradient(pred, model.trainable_variables)
        # concatenate the gradients of all trainable variables,
        # not discriminating between weights and biases
        g = tf.concat([tf.reshape(i, shape=(-1,1)) for i in g], axis=0)
        grad.append(g)

    grad = tf.concat(grad,axis=1)
    ntk = tf.einsum('ij,ik->jk', grad, grad)
    return ntk


def f_eig(t, model, input, order, learning_rate):
  ntk = compute_ntk(model, input).numpy()
  ntk = 0.5 * (ntk.T + ntk)
  f0 = model(x)
  f0 = f0.numpy().reshape((f0.numpy().shape[0],))
  
  eigval, eigvec = np.linalg.eig(ntk)

  f0_tilde = [np.dot(f0, eigvec[:, k]) for k in range(eigval.size)]
  y_tilde = [np.dot(y_noisy, eigvec[:, k]) for k in range(eigval.size)]
  
  aux = np.zeros_like(eigvec[:,0])
  for k in range(order):
    coeff1 = f0_tilde[k] 
    coeff2 = (1 - np.exp(- eigval[k] * learning_rate * t) ) * (y_tilde[k] - f0_tilde[k])
    aux = np.add(aux, (coeff1 + coeff2) * eigvec[:,k])

  return aux, f0


if __name__ == "__main__":
    # Step 1: Generate Data
  np.random.seed(42)  # For reproducibility
  x = np.linspace(-1, 1, 100)  # Input data
  test_function = scipy.special.legendre(3)
  y_true = test_function(x) #np.sin(x)  # True function
  noise = np.random.normal(0, 0.1, size=y_true.shape)  # Adding some noise
  y_noisy = y_true #+ noise  # Noisy target data

  # Step 2: Create a Neural Network for Regression
  initialiser = tf.keras.initializers.HeNormal(seed=None)
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(28, activation='tanh', input_shape=(1,), kernel_initializer=initialiser),
      tf.keras.layers.Dense(20, activation='tanh', kernel_initializer=initialiser),
      tf.keras.layers.Dense(1,kernel_initializer=initialiser)  # Output layer
  ])


  fig, ax = plt.subplots()
  ft, f0 = f_eig(0, model, x, 100, 0.001)
  line = ax.plot(x, ft, color='orange', label='eigenspace')

  ax.scatter(x, y_noisy, label="Noisy Data", color='red', alpha=0.4)
  def update(t):
      # for each frame, update the data stored on each artist.
      y, _ = f_eig(t, model, x, 100, 0.001)
      ax.set_title(f"t = {t}")
      # update the scatter plot:
      # update the line plot:
      line[0].set_ydata(y)
      return line

  ani = animation.FuncAnimation(fig=fig, func=update, frames=1000, interval=1)

  writervideo = animation.FFMpegWriter(fps=60) 
  ani.save('integrated_solution.mp4', writer=writervideo) 
  plt.close() 