# Copyright (c) 2012 Boris Reuderink
# License: BSD
import logging
import numpy as np
from scipy import signal


log = logging.getLogger(__name__)


def windows(indices, offset, X):
  '''Cut windows specified by indices from X with offsets.

  Cut windows from a multivariate time series. This can be used to
  extract a sliding window over the data, or the extract trials for
  neurophysiological experiments from a continuous recording.

  Parameters
  ----------
  indices : list or array with ints
      The indices of X that are of interest for the window. The
      indices do not necessarily have to indicate the begin or center
      of the window, since offsets relative to the indices determine
      the window placement.
  offset : tuple with ints (start, end)
      The start and end of the window relative to the index of the
      event. A negative start can be used for example to extract a
      baseline before an event.
  X : 2D array-like
      The continuous array from which windows of the horizontal axis
      are selected.
  
  Returns
  -------
  out : nd_array
    A 3D array, with the first dimension indexing the different
    windows.

  Examples
  --------
  >>> X = np.arange(60).reshape(2, 30)
  >>> T = windows([10, 15, 20], [-2, 4], X)
  >>> T.shape
  (3, 2, 6)

  >>> T[0]
  array([[  8.,   9.,  10.,  11.,  12.,  13.],
         [ 38.,  39.,  40.,  41.,  42.,  43.]])
  '''

  # construct vector y and tensor T
  indices = np.atleast_1d(indices)
  start, end = offset

  T = np.zeros((indices.size, X.shape[0], end-start)) * np.nan
  for ti, i in enumerate(indices):
    T[ti] = X[:,(i+start):(i+end)]

  return T


def spec(T, axis=0):
  '''Combined detrending, windowing and real-valued FFT over axis.

  Usually, the FFT is preceded by a detrending and windowing to reduce
  edge effects. This function performs the detrending, windowing and
  real-valued FFT.

  Parameters
  ----------
  T : ndarray 
      Tensor containing data to be detrended.
  axis : int, optional
      Specifies the axis over which the spectrum is calculated.

  Returns
  -------
  out: ndarray
    Array similar to `T` where the time dimension is replace with
    it's frequency spectrum.
  '''
  T = signal.detrend(T, axis=axis)
  win = np.hanning(T.shape[axis])
  win.shape = (np.where(np.arange(T.ndim) == axis, -1, 1))
  return np.fft.rfft(T * win, axis=axis)


def spec_weight(freqs, hp=None, lp=None, bleed=7):
  '''Construct weight vector for filtering in the frequency domain.

  A simple spectral filter can be implemented by weighting the
  spectrum of a signal. This function returns a vector with spectrum
  weights. 

  Parameters
  ----------
  freqs : array-like
    The frequency of the DFT bins.
  fs : int
    The sampling rate of the window in Hz.
  hp : float, optional
    The cut-off value in Hz for the low-pass filter.
  lp : float, optional
    The cut-off value in Hz for the high-pass filter.
  bleed : int, optional
    The length of the Hanning window that is convolved with the
    brick-wall filter to smooth the filter response.

  Returns
  -------
  out : ndarray
    Vector with weights for increasing frequencies, to be multiplied
    with real-valued DFT-ed data.

  Examples
  --------
  >>> assert False
  
  '''
  freqs = np.abs(freqs)  # Make filter symmetrical for negative frequencies.

  # construct brick-wall response
  response = np.ones(freqs.size)
  if lp:
    response[freqs > lp] = 0.
  if hp:
    response[freqs < hp] = 0.

  # smooth spectrum to prevent ringing
  win, pad = np.hanning(bleed), np.ones(bleed)
  padded = np.hstack([pad * response[0], response, pad * response[-1]])
  response = np.convolve(padded, win/np.sum(win), 'same')
  return response[pad.size:-pad.size]


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
