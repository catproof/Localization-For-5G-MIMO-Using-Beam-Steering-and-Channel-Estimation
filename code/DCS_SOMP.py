import numpy as np
import numpy.matlib

from util import norm_2, print_debug

def modified_DCS_SOMP(
  Y: np.ndarray(shape=[640, 10], dtype="complex_"), 
  A: np.ndarray(shape=[640, 4960, 10], dtype="complex_"), 
  L: np.double, 
):
  """
  Y = input (one column vector for each channel / subcarrier)
  A = sensing matrix for each channel
  L = the sparsity level
  """
  K: int = Y.shape[1]     # number of channels
  N: int = Y.shape[0]     # observation per channel
  M: int = A.shape[1]     # size of sparse vector

  L = N if (L <= 0) else L

  R: np.ndarray = Y.copy()
  psi: np.ndarray = np.zeros([N, L, K], dtype="complex_")
  indices: np.ndarray = np.zeros([L])
  columns: np.ndarray = np.zeros([N, L, K], dtype="complex_")
  betamatrix: np.ndarray = np.zeros([L, K], dtype="complex_")

  # TODO: in here do the same as adding temp variables, check the dimension and any unreal big/small numbers in arrays
  for c in range(0, L):
    cost: np.ndarray = np.zeros([M])
    for m in range(0, M):
      for k in range(0, K):
        cost[m] += abs(A[:, m, k].conjugate().transpose().dot(R[:, k])) / norm_2(A[:, m, k])

    max_index = np.argmax(cost)
    indices[c] = max_index

    for k in range(0, K):
      columns[:, c, k] = A[:, max_index, k]
      omega = A[:, max_index, k]
      psi[:, c, k] = omega
      for c2 in range(0, c-1):
        psi[:, c, k] -= psi[:, c2, k].conjugate().transpose().dot(omega) * psi[:, c2, k] / np.power(norm_2(psi[:, c2, k]), 2)
      
      beta = psi[:, c, k].conjugate().transpose().dot(R[:, k]) / np.power(norm_2(psi[:, c, k]), 2)
      betamatrix[c, k] = beta
      R[:, k] -= beta * psi[:, c, k]

  h: np.ndarray = np.zeros([L, K], dtype="complex_")
  for k in range(0, K):
    Q, Rqr = np.linalg.qr(columns[:, :, k])
    h[:, k] = np.linalg.inv(Rqr).dot( Q.conjugate().transpose()).dot(psi[:, :, k]).dot(betamatrix[:, k])
  
  return indices, h
