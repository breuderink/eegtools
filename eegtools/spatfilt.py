import numpy as np


def car(n):
  '''
  Return a common average reference (CAR) spatial filter for n channels.

  The common average reference is a re-referencing scheme that is
  commonly used when no dedicated reference is given. Since the
  average signal is subtracted from each sensor's signal, it reduces
  signals that are common to all sensors, such as far-away noise.

  Parameters
  ----------
  n : int
    The number of sensors to filer.

  Returns
  -------
  W : array
    Spatial filter matrix of shape (n, n) where n is the number of
    sensors. Each row of W is single spatial filter.

  Examples
  --------
  >>> car(4)
  array([[ 0.75, -0.25, -0.25, -0.25],
         [-0.25,  0.75, -0.25, -0.25],
         [-0.25, -0.25,  0.75, -0.25],
         [-0.25, -0.25, -0.25,  0.75]])

  '''
  return np.eye(n) - 1. / float(n)


def select_channels(n, keep_inds):
  '''
  Return a spatial filter to select specific channels.

  The selection of a subset of channels can be combined with other
  spatial filters with a matrix multiplication.

  Parameters
  ----------
  n : int
    The number of sensors.
  keep_inds : array with indices or bools
    Contains the indices to keep; either as a boolean mask or as
    indices.

  Returns
  -------
  W : 2D array
    Spatial filter matrix with spatial filters in rows. W has shape
    (n, o) where n is the number of sensors, and o is the number of
    selected filters.

  Examples
  --------
  >>> select_channels(4, [2, 3])
  array([[ 0.,  0.,  1.,  0.],
         [ 0.,  0.,  0.,  1.]])

  >>> select_channels(4, np.array([True, False, True, False]))
  array([[ 1.,  0.,  0.,  0.],
         [ 0.,  0.,  1.,  0.]])
  '''
  return np.eye(n)[keep_inds]


def whitener(C, rtol=1e-15):
  '''
  Calculate the whitening transform for signals with covariance C.

  The whitening transform is used to remove covariance between
  signals, and can be regarded as principal component analysis with
  rescaling and an optional rotation. The whitening transform is
  calculated as C^{-1/2}, and implemented to work with rank-deficient
  covariance matrices.

  Parameters
  ----------
  C : array-like, shape (p, p)
    Covariance matrix of the signals to be whitened.
  rtol : float, optional
    Cut-off value specified as the fraction of the largest eigenvalue
    of C.

  Returns
  -------
  W : array, shape (p, p)
    Symmetric matrix with spatial filters in the rows.

  See also
  --------
  car : common average reference

  Examples
  --------
  >>> X = np.random.randn(3, 100) 
  >>> W = whitener(np.cov(X))
  >>> X_2 = np.dot(W, X - np.mean(X, axis=1).reshape(-1, 1))

  The covariance of X_2 is now close to the identity matrix:
  >>> np.linalg.norm(np.cov(X_2) - np.eye(3)) < 1e-10
  True
  '''
  e, E = np.linalg.eigh(C)
  return reduce(np.dot, 
    [E, np.diag(np.where(e > np.max(e) * rtol, e, np.inf)**-.5), E.T])


def outer_n(n):
  '''
  Return a list with indices from both ends. Used for CSP.
 
  Parameters
  ----------
  n : int
    The number of indices to select.

  Returns
  -------
  out : array
    Contains the indices picked from both ends.

  See also
  --------
  csp : common spatial patterns algorithm.

  Examples
  --------
  >>> outer_n(6)
  array([ 0,  1,  2, -3, -2, -1])

  '''
  return np.roll(np.arange(n) - n/2, (n + 1) / 2)


def csp_base(C_a, C_b):
  '''
  Calculate the full common spatial patterns (CSP) transform. 

  The CSP transform finds spatial filters that maximize the variance
  in one condition, and minimize the signal's variance in the other.
  See [1]. Usually, only a subset of the spatial filters is used.

  Parameters
  ----------
  C_a : array-like of shape (n, n)
    Sensor covariance in condition 1.
  C_b : array-like of shape (n, n)
    Sensor covariance in condition 2.

  Returns
  -------
  W : array of shape (m, n)
    A matrix with m spatial filters with decreasing variance in the
    first condition. The rank of (C_a + C_b) is determines the number
    of filters m.

  See also
  --------
  csp : common spatial patterns algorithm.
  outer_n : pick indices from both sides.

  References
  ----------
  [1] Zoltan J. Koles. The quantitative extraction and topographic
  mapping of the abnormal components in the clinical EEG.
  Electroencephalography and Clinical Neurophysiology,
  79(6):440--447, December 1991.


  Examples:
  ---------
  In condition 1 the signals are positively correlated, in condition 2
  they are negatively correlated. Their variance stays the same:
  >>> C_1 = np.ones((2, 2))
  >>> C_1
  array([[ 1.,  1.],
         [ 1.,  1.]])

  >>> C_2 = 2 * np.eye(2) - np.ones((2, 2))
  >>> C_2
  array([[ 1., -1.],
         [-1.,  1.]])

  The most differentiating projection is found with the CSP transform:
  >>> csp_base(C_1, C_2).round(2)
  array([[-0.5,  0.5],
         [ 0.5,  0.5]])
  '''
  P = whitener(C_a + C_b)
  P_C_b = reduce(np.dot, [P, C_b, P.T])
  _, _, B = np.linalg.svd((P_C_b))
  return np.dot(B, P.T)


def csp(C_a, C_b, m):
  '''
  Calculate common spatial patterns (CSP) transform. 

  The CSP transform finds spatial filters that maximize the variance
  in one condition, and minimize the signal's variance in the other.
  See [1]. Usually, only a subset of the spatial filters is used.

  Parameters
  ----------
  C_a : array-like of shape (n, n)
    Sensor covariance in condition 1.
  C_b : array-like of shape (n, n)
    Sensor covariance in condition 2.
  m : int
    The number of CSP filters to extract.

  Returns
  -------
  W : array of shape (m, n)
    A matrix with m/2 spatial filters that maximize the variance in
    one condition, and m/2 that maximize the variance in the other.

  See also
  --------
  csp_base : full common spatial patterns transform.
  outer_n : pick indices from both sides.

  References
  ----------
  [1] Zoltan J. Koles. The quantitative extraction and topographic
  mapping of the abnormal components in the clinical EEG.
  Electroencephalography and Clinical Neurophysiology,
  79(6):440--447, December 1991.

  Examples:
  ---------
  We construct two nearly identical covariance matrices for condition
  1 and 2:
  >>> C_1 = np.eye(4)
  >>> C_2 = np.eye(4)
  >>> C_2[1, 3] = 1

  The difference between the conditions is in the 2nd and 4th sensor:
  >>> C_2 - C_1
  array([[ 0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  1.],
         [ 0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.]])

  The two most differentiating projections are found with the CSP transform.
  Indeed, it projects the same sensors:
  >>> csp(C_1, C_2, 2).round(2)
  array([[ 0.  ,  0.37,  0.  ,  0.6 ],
         [ 0.  , -0.6 ,  0.  ,  0.37]])
  '''
  W = csp_base(C_a, C_b)
  assert W.shape[1] >= m
  return W[outer_n(m)]
