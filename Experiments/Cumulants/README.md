# List of results and discussion

The jupyter notebook `cumulants.ipynb` computes the fourth empirical cumulant of the outputs of an ensemble of neural networks. It is possible to set the architecture, the activation functions, the number of replicas, and the output function. The initialisation of the parameters can be modified by tweaking the model generator.

## General technical information

+ Neural network models are generated using `tf.keras.models.Sequential`, using `tf.keras.layers.Dense` layers.

+ As default, weights are initialised using `GlorotNormal` (as in NNPDF).

+ Biases are initialised to zero.

+ All generators are seeded, meaning that experiments are reproducible (at least with the same machine).

+ The fourth cumulant $k_4$ is  a four-rank tensor, defined as follows
$$
k_{4, \alpha_1 \alpha_2 \alpha_3 \alpha_4} =
\mathbb{E} \left[
  f_{\alpha_1}
  f_{\alpha_2}
  f_{\alpha_3}
  f_{\alpha_4}
\right] -
  \mathbb{E} \left[
  f_{\alpha_1}
  f_{\alpha_2}
  \right]
  \mathbb{E} \left[
  f_{\alpha_3}
  f_{\alpha_4}
  \right] -
  \mathbb{E} \left[
  f_{\alpha_1}
  f_{\alpha_3}
  \right]
  \mathbb{E} \left[
  f_{\alpha_2}
  f_{\alpha_4}
  \right] -
  \mathbb{E} \left[
  f_{\alpha_1}
  f_{\alpha_4}
  \right]
  \mathbb{E} \left[
  f_{\alpha_2}
  f_{\alpha_3}
  \right] \,,
$$
where $f$ indicates the output of the neural network and $f_{\alpha} \equiv f(x_{\alpha})$ with $\alpha = 1,\dots, N_{\textrm{dat}}$. The expectation values are computed empirically, namely computing the empirical statistics out of the ensemble of networks
$$
\mathbb{E}\left[ F (f,\dots) \right] \approx
\frac{1}{N_{\textrm{rep}}}
\sum_{k=0}^{N_{\textrm{rep}}}
  F^{(k)} (f,\dots) \,,
$$
where $F$ is a generic function of the network outputs and $F^{(k)}$ is the function computed with the $k$-th neural network replica. For example

$$\mathbb{E}\left[
  f_{\alpha_1}
  f_{\alpha_2}
  f_{\alpha_3}
  f_{\alpha_4}
\right] \approx
\frac{1}{N_{\textrm{rep}}}
\sum_{k=0}^{N_{\textrm{rep}}}
  f_{\alpha_1}^{(k)}
  f_{\alpha_2}^{(k)}
  f_{\alpha_3}^{(k)}
  f_{\alpha_4}^{(k)} \,.$$
  
Here the output of the neural network is assumed to be one-dimensional, but an additional neural index is possible for each neural output.


## List of experiments
The following list contains the experiment that have been carried out, together with the results.

### Experiments 19^th^ Sep
These experiments use the following settings
| Seed | Weight initialisation | Bias initialisation | Activation function | Output function | Number of points |
|-|-|-|-|-| - |
|32152315| `GlorotNormal` | `Zeros` | Tanh | Linear | 150 |

+ Using the NNPDF architecture [1,25,20,8]:
    | | Output 1 | Output 2 | Output 3 | Output 4 | Output 5 | Output 6 | Output 7 | Output 8
    |-|-|-|-|-|-|-|-|-|
    sup($k4$) | -3 $\times$  | 0.006 | 0.005 | 0.004 | 0.003 | 0.0 | 0.0 | 0.0
    min($k4$) | -0.0003 | 0.0 | 0.0 | 0.0 |  0.0 | -0.006 | -0.003 | -0.003

+ Using NNPDF with $(x, log(x))$ [2, 25, 20, 8]
    | | Output 1 | Output 2 | Output 3 | Output 4 | Output 5 | Output 6 | Output 7 | Output 8
    |-|-|-|-|-|-|-|-|-|
    sup($k4$) |  |  |  |  |  |  |  |  |
    min($k4$) |  |  |  |  |  |  |  |  |

+ Using a large network [1,100,100,1]
    | | Output 1 | 
    |-| -------- |
    | sup($k4$) | $6.75 \times 10^{-24}$ |
    | min($k4$) | $- 6 \times 10^{-4}$ |

