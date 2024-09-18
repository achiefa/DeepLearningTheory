from numba import jit
import numpy as np

def compute_m1(nn_outputs):
  return nn_outputs.mean(axis=0)


def compute_m2(nn_outputs):
  number_of_replica = nn_outputs.shape[0]
  x_size = nn_outputs.shape[1]
  out_size = nn_outputs.shape[2]
  result = np.zeros((x_size, out_size, x_size, out_size))
  for rep in range(number_of_replica):
    result += np.multiply.outer(nn_outputs[rep], nn_outputs[rep])
  result /= number_of_replica
  return result


def compute_m4(nn_outputs, out_id=0):
  number_of_replica = nn_outputs.shape[0]
  x_size = nn_outputs.shape[1]
  result = np.zeros((x_size, x_size, x_size, x_size))
  for rep in range(number_of_replica):
    aux = np.multiply.outer(nn_outputs[rep][:,out_id], nn_outputs[rep][:,out_id])
    result += np.multiply.outer(aux,aux)
  result /= number_of_replica
  return result


#@jit(nopython=True)
#def compute_k2(nn_outputs, out_idx1, out_idx2):
#  x_size = nn_outputs.shape[1]
#  m1 = compute_m1(nn_outputs)
#  m2 = compute_m2(nn_outputs)[:,out_idx1,:,out_idx2]
#  result = np.zeros_like(m2)
#  for a1 in range(x_size):
#    for a2 in range(x_size):
#      result[a1,a2] = m2[a1,a2] - m1[a1] * m1[a2]
#  del m1, m2
#  return result


@jit(nopython=True)
def compute_k4(m2_sliced, m4_sliced, x_size):
  result = np.zeros_like(m4_sliced)
  for a1 in range(x_size):
    for a2 in range(x_size):
      for a3 in range(x_size):
        for a4 in range(x_size):
          result[a1,a2,a3,a4] = m4_sliced[a1,a2,a3,a4] - m2_sliced[a1,a2] * m2_sliced[a3, a4] - m2_sliced[a1,a3] * m2_sliced[a2, a4] - m2_sliced[a1, a4] * m2_sliced[a2, a3]
  return result



@jit(nopython=True)
def compute_m2_test(nn_outputs, out=0):
  number_of_replica = nn_outputs.shape[0]
  x_size = nn_outputs.shape[1]
  result = np.zeros((x_size, x_size))
  for rep in range(number_of_replica):
    for a1 in range(x_size):
      for a2 in range(x_size):
        result[a1,a2] += nn_outputs[rep][a1][0] * \
                         nn_outputs[rep][a2][0]
  result /= number_of_replica
  return result

@jit(nopython=True)
def compute_m4_test(nn_outputs, out_idx):
  number_of_replica = nn_outputs.shape[0]
  x_size = nn_outputs.shape[1]
  result = np.zeros((x_size, x_size, x_size, x_size))
  for rep in range(number_of_replica):
    for a1 in range(x_size):
      for a2 in range(x_size):
        for a3 in range(x_size):
          for a4 in range(x_size):
            result[a1,a2,a3,a4] += nn_outputs[rep][a1][out_idx] * \
                                   nn_outputs[rep][a2][out_idx] * \
                                   nn_outputs[rep][a3][out_idx] * \
                                   nn_outputs[rep][a4][out_idx]
  result /= number_of_replica
  return result
