# List of results and discussion

The jupyter notebook `cumulants.ipynb` computes the fourth cumulant of the outputs of an ensemble of neural networks. It is possible to set the architecture, the activation functions, the number of replicas, and the output function. The initialisation of the parameters can be modified by tweaking the model generator.

## General technical information

+ Neural network models are generated using `tf.keras.models.Sequential`, using `tf.keras.layers.Dense` layers.

+ The fourth cumulant $k_4$
