import numpy as np

# The largest xgrid used NNPDF
XGRID = np.array([2.00000000e-07, 3.03430477e-07, 4.60350147e-07, 6.98420853e-07,
       1.05960950e-06, 1.60758550e-06, 2.43894329e-06, 3.70022721e-06,
       5.61375772e-06, 8.51680668e-06, 1.29210157e-05, 1.96025050e-05,
       2.97384954e-05, 4.51143839e-05, 6.84374492e-05, 1.03811730e-04,
       1.57456056e-04, 2.38787829e-04, 3.62054496e-04, 5.48779532e-04,
       8.31406884e-04, 1.25867971e-03, 1.90346340e-03, 2.87386758e-03,
       4.32850064e-03, 6.49620619e-03, 9.69915957e-03, 1.43750686e-02,
       2.10891867e-02, 3.05215840e-02, 4.34149174e-02, 6.04800288e-02,
       8.22812213e-02, 1.09143757e-01, 1.41120806e-01, 1.78025660e-01,
       2.19504127e-01, 2.65113704e-01, 3.14387401e-01, 3.66875319e-01,
       4.22166775e-01, 4.79898903e-01, 5.39757234e-01, 6.01472198e-01,
       6.64813948e-01, 7.29586844e-01, 7.95624252e-01, 8.62783932e-01,
       9.30944081e-01, 1.00000000e+00])


def round_float(value, ref_value, tol_magnitude=1.e-6):
  """
  Utility function used to round to zero according to a reference value.
  """
  ratio = abs(value) / abs(ref_value)
  if ratio < tol_magnitude:
    return 0.0
  else:
    return value


def extract_independent_columns(matrix, DEBUG=False, **kwargs):
    """
    Extract a subset of linearly independent columns from the matrix, 
    preserving the original order.
    
    Args:
        matrix (numpy.ndarray): The input square matrix.
        
    Returns:
        numpy.ndarray: A matrix containing only the independent columns.
    """
    independent_columns = []
    current_matrix = np.empty((matrix.shape[0], 0), dtype=matrix.dtype)

    # Starting condition
    current_matrix = np.hstack([current_matrix, matrix[:, 0 : 0 + 1]])

    for col_idx in range(1, matrix.shape[1]):
        candidate_matrix = np.hstack([current_matrix, matrix[:, col_idx : col_idx + 1]])
        if np.linalg.matrix_rank(candidate_matrix, **kwargs) > np.linalg.matrix_rank(current_matrix, **kwargs):
            independent_columns.append(col_idx)
            current_matrix = candidate_matrix

    return matrix[:, independent_columns], independent_columns



from n3fit.layers import FlavourToEvolution, FkRotation

# Flavour map for NNPDF rotation
flavour_map = [
        {'fl': 'u'},
        {'fl': 'ubar'},
        {'fl': 'd'},
        {'fl': 'dbar'},
        {'fl': 's'},
        {'fl': 'sbar'},
        {'fl': 'c'},
        {'fl': 'g'},
    ]

# Particle ID map
PID_map = {
  'd'    : 1,
  'u'    : 2,
  's'    : 3,
  'c'    : 4,
  'b'    : 5,
  'dbar' : -1,
  'ubar' : -2,
  'sbar' : -3,
  'cbar' : -4,
  'bbar' : -5,
  'g'    : 21
}

fk_ev_map = [
  r'$\Sigma$',
  r'$g$',
  r'$V$',
  r'$V_3$',
  r'$V_8$',
  r'$T_3$',
  r'$T_8$',
  r'$c^+$',
  r'$V_{15}$',
]

def produce_R_flav_8_to_ev_9():
  # Rotates from flavour basis to evolution basis.
  # The evolution basis is [sng, g, v, v3, v8, t3, t8, cp, v15]
  # (v_ev)_j = R_{ij} * (v_fl)_{i} ---> (v_ev) = R^T * v_fl
  R_flav_8_to_ev_9_layer = FlavourToEvolution(flavour_map, "FLAVOUR")
  return R_flav_8_to_ev_9_layer.rotation_matrix.numpy()   

def produce_R_ev_9_to_flav_8():
  R_flav_8_to_ev_9 = produce_R_flav_8_to_ev_9()
  # Invert rotation matrix
  RR_T = R_flav_8_to_ev_9 @ R_flav_8_to_ev_9.T
  RR_inv = np.linalg.inv(RR_T)
  return RR_inv @ R_flav_8_to_ev_9