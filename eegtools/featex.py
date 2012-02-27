# Copyright (c) 2012 Boris Reuderink
# License: BSD
import logging
import numpy as np
from scipy import signal


log = logging.getLogger(__name__)


def windows(indices, offset, X):
  '''Cut windows specified by indices from X with offsets.'''
  # construct vector y and tensor T
  indices = np.atleast_1d(indices)
  start, end = offset

  T = np.zeros((indices.size, X.shape[0], end-start)) * np.nan
  for ti, i in enumerate(indices):
    T[ti] = X[:,(i+start):(i+end)]

  return T


def spec(T, axis=0):
  '''Combined detrending, windowing and real-valued FFT over axis.'''
  T = signal.detrend(T, axis=axis)
  win = np.hanning(T.shape[axis])
  win.shape = (np.where(np.arange(T.ndim) == axis, -1, 1))
  return np.fft.rfft(T * win, axis=axis)


def spec_weight(n, fs, band, bleed=7):
  '''Construct weight vector for FFT-based filtering.'''
  # construct brick-wall response
  freq = np.fft.fftfreq(n, 1./fs)
  response = np.where(np.logical_and(freq >= band[0], freq < band[1]), 1, 0)

  # smooth spectrum to prevent ringing
  win = np.hanning(bleed)
  return np.convolve(response, win/np.sum(win), 'same')


def band_cov(bf):
  '''
  Rank-2 covariance matrix for matching DFT coefficients of sensors. Used in
  covtens().
  '''
  return np.real(np.outer(bf, bf.conj()))


def cov_tens(T):
  '''
  Covariance tensor for real DFT-ed trial (p x n) matrix T. An (n x p x p)
  tensor C is returned. The idea of constructing a tensor with covariance
  matrices for classification was presented in [1].

  Please note that T[0,:,:] is the covariance of the DC offset.

  >>> p, n = 10, 128
  >>> X = np.random.randn(p, n)
  >>> T = cov_tens(np.fft.fft(X, axis=1))
  >>> Sig = np.sum(T[1:], 0)
  >>> C = np.cov(X, bias=True)
  >>> np.testing.assert_almost_equal(Sig, C)

  [1] Jason Farquhar. A linear feature space for simultaneous learning of
  spatio-spectral filters in BCI. Neural Networks, 22:1278--1285, 2009.
  '''
  T = np.atleast_2d(T)
  C = np.asarray([band_cov(T[:, i]) for i in range(T.shape[1])])
  return C / T.shape[1]**2


def whitener(Sigma, rtol=1e-15):
  '''
  Calculate whitening transform \Sigma^{-1/2}. Works with rank-deficient
  covariance matrices.
  '''
  e, E = np.linalg.eigh(Sigma)
  return reduce(np.dot, 
    [E, np.diag(np.where(e > np.max(e) * rtol, e, np.inf)**-.5), E.T])


def tile_tens3d(T, width=None):
  '''
  Tile the slabs of a 3D tensor T. Useful for visualization of stacks of
  arrays. The width parameter determines the width of the tile grid.

  >>> tile_tens3d(np.zeros((5, 2, 3)) + np.arange(5).reshape(-1, 1, 1))
  array([[  0.,   0.,   0.,   1.,   1.,   1.,   2.,   2.,   2.],
         [  0.,   0.,   0.,   1.,   1.,   1.,   2.,   2.,   2.],
         [  3.,   3.,   3.,   4.,   4.,   4.,  nan,  nan,  nan],
         [  3.,   3.,   3.,   4.,   4.,   4.,  nan,  nan,  nan]])
  '''
  T = np.atleast_3d(T)
  n, ch, cw = T.shape
  
  # calculate grid shape
  w = width if width else np.ceil(np.sqrt(n))
  h = np.ceil(n/float(w))

  # construct 2D tile matrix
  R = np.vstack([T, np.nan * np.zeros((w*h - n, ch, cw))])
  R = R.reshape(h, w, ch, cw)
  R = np.concatenate(R, axis=1)
  R = np.concatenate(R, axis=1)
  return R
