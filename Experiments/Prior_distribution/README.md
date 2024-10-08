# Results and discussion

The jupyter notebook `prior.ipynb` plots the functional prior induced by the neural network at initialisation. The prior is shown with mean value and uncertainty bands, both computed using the empirical statistics of the ensemble.

The weights in each layer are initialised using `GlorotNormal`, while the biases are initialised to zero. The activation function for all layers is Tanh, except for the last layer (output) which is linear. A seed is also used to reproduce all the initialised parameters and the results. Note that different initialisations are possible, provided that the function that creates the model is modified.

## Plot as function of the architecture
The following plots show the network output as function of the input with central value (thick line) and uncertainty bands corresponding to the 1-$\sigma$ interval (shaded area) and the 68% confidence level (dashed line).

These plots have been obtained with the following settings:

| Seed | Activation function | Output function | Weight initialisation | Bias initialisation | Number of points |
| - | - | - | - | - | - |
| 32152315 | Tanh | Linear | `GloriotNormal` | `Zeros` | 1000 |

<div style="text-align: center;">
  <img src="gloriot_normal.png" alt="description" width="70%">
</div>



## Plot of the functional prior with NNPDF-like network
These plots have been obtained with the following settings:

| Seed | Activation function | Output function | Weight initialisation | Bias initialisation | Number of points |
| - | - | - | - | - | - |
| 32152315 | Tanh | Linear | `GloriotNormal` | `Zeros` | 1000 |

### Priors with one dimensional outputs
Plots with one-dimensional input $(x)$.

<div style="text-align: center;">
  <img src="nnpdf_gloriot_normal.png" alt="description" width="70%">
</div>

### Priors including $\log (x)$ in the input

Plots with two-dimensional inputs $(x, \log(x))$.

<div style="text-align: center;">
  <img src="nnpdf_log_gloriot_normal.png" alt="description" width="70%">
</div>

# NNPDF prior
## Imposing sum rules and pre-processing

<div style="text-align: center;">
  <img src="nnpdf_prior.png" alt="description" width="70%">
</div>

## Imposing sum rules solely
<div style="text-align: center;">
  <img src="nnpdf_prior_no_sumrule.png" alt="description" width="70%">
</div>

## Imposing pre-processing solely

<div style="text-align: center;">
  <img src="nnpdf_prior_no_pre_proc.png" alt="description" width="70%">
</div>

## Relaxing pre-processing and sum rules

<div style="text-align: center;">
  <img src="nnpdf_prior_no_sumrule_no_pre_proc.png" alt="description" width="70%">
</div>