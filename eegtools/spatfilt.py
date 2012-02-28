import numpy as np
def whitener(Sigma, rtol=1e-15):
  '''
  Calculate whitening transform \Sigma^{-1/2}. Works with rank-deficient
  covariance matrices.
  '''
  e, E = np.linalg.eigh(Sigma)
  return reduce(np.dot, 
    [E, np.diag(np.where(e > np.max(e) * rtol, e, np.inf)**-.5), E.T])


