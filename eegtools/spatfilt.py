import numpy as np


def car(n):
  '''Return a common average reference spatial filter for n channels'''
  return np.eye(n) - 1. / float(n)


def select_channels(n, keep_inds):
  '''
  Spatial filter to select channels keep_inds out of n channels. 
  Keep_inds can be both a list with indices, or an array of type bool.
  '''
  return np.eye(n)[:, keep_inds]


def whitener(Sigma, rtol=1e-15):
  '''
  Calculate whitening transform \Sigma^{-1/2}. Works with rank-deficient
  covariance matrices.
  '''
  e, E = np.linalg.eigh(Sigma)
  return reduce(np.dot, 
    [E, np.diag(np.where(e > np.max(e) * rtol, e, np.inf)**-.5), E.T])


def outer_n(n):
  '''Return a list with indices from both ends, i.e.: [0, 1, 2, -3, -2, -1]'''
  return np.roll(np.arange(n) - n/2, (n + 1) / 2)


def csp_base(sigma_a, sigma_b):
  '''Return CSP transformation matrix. No dimension reduction is performed.'''
  P = whitener(sigma_a + sigma_b)
  P_sigma_b = reduce(np.dot, [P, sigma_b, P.T])
  _, _, B = np.linalg.svd((P_sigma_b))
  return np.dot(B, P.T)


def csp(sigma_a, sigma_b, m):
  '''
  Return a CSP transform for the covariance for class a and class b,
  with the m outer (~discriminating) spatial filters (if they exist).
  '''
  W = csp_base(sigma_a, sigma_b)
  if W.shape[1] > m: 
    return W[:, outer_n(m)]
  return W
