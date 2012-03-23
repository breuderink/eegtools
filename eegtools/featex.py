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
    baseline before an event. If a window is placed partially outside
    of X, this window is ignored.
  X : 2D array-like
    The continuous array from which windows of the horizontal axis
    are selected.

  Returns
  -------
  W : nd_array
    A 3D array, with the first dimension indexing the different
    windows.
  indices : array of ints
    The subset of indices that point to valid windows.

  Examples
  --------
  >>> X = np.arange(60).reshape(2, 30)
  >>> T, ii  = windows([0, 10, 15, 20], [-2, 4], X)
  >>> T.shape
  (3, 2, 6)

  >>> ii
  array([10, 15, 20])

  >>> T[0]
  array([[  8.,   9.,  10.,  11.,  12.,  13.],
         [ 38.,  39.,  40.,  41.,  42.,  43.]])
  '''

  # construct vector y and tensor T
  X = np.atleast_2d(X)
  p, n = X.shape
  ii = np.atleast_1d(indices)
  begin, end = offset

  ii = ii[np.logical_and(ii + begin >= 0, ii + end < n)]

  W = np.zeros((ii.size, p, end - begin)) * np.nan
  for ti, i in enumerate(ii):
    W[ti] = X[:, i + begin:i + end]

  return W, ii


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

  See also:
  ---------
  windows : extract windows from continuous data.
  spec_weight : filter in the frequency domain.
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
    The frequency of the DFT bins, for example from numpy.fft.fftfreq
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

  See also:
  ---------
  windows : extract windows from continuous data.
  spec : calculate power spectrum.

  Examples
  --------
  >>> freq = np.fft.fftfreq(32, d=1./32)  # get freqency domain for w
  >>> w = spec_weight(freq, lp=2.)        # calculate low-pass response

  >>> np.vstack([freq, w]).T              # doctest: +ELLIPSIS
  array([[  0.        ,   1.        ],
         [  1.        ,   0.91666667],
         [  2.        ,   0.66666667],
         [  3.        ,   0.33333333],
         [  4.        ,   0.08333333],
         [  5.        ,   0.        ],
         ...
         [ -6.        ,   0.        ],
         [ -5.        ,   0.        ],
         [ -4.        ,   0.08333333],
         [ -3.        ,   0.33333333],
         [ -2.        ,   0.66666667],
         [ -1.        ,   0.91666667]])

  '''
  freqs = np.abs(freqs)  # Make filter symmetrical for negative frequencies.
  bleed = int(bleed)

  # construct brick-wall response
  response = np.ones(freqs.size)
  if lp:
    response[freqs > float(lp)] = 0.
  if hp:
    response[freqs < float(hp)] = 0.

  # smooth spectrum to prevent ringing
  win, pad = np.hanning(bleed), np.ones(bleed)
  padded = np.hstack([pad * response[0], response, pad * response[-1]])
  response = np.convolve(padded, win / np.sum(win), 'same')
  return response[pad.size:-pad.size]


def band_cov(bf):
  '''
  Rank-2 covariance matrix for matching DFT coefficients of sensors.
  Used in covtens().

  Parameters
  ----------
  bf : complex-valued array
    Entries of `bf` represent different channels, and hold a single,
    complex-valued frequency bin as generated with the DFT.

  Returns
  -------
  out : 2D real-valued array
    Channel-covariance matrix, containing the covariance within the
    DFT frequency bin for pairs of channels.

  See also
  --------
  cov_tens : calculate covariance tensor for DFT-ed windows.
  '''
  return np.real(np.outer(bf, bf.conj()))


def cov_tens(T):
  '''
  Calculate per-frequency covariance.

  The idea of constructing a tensor with covariance matrices for
  classification was presented in [1]. Please note that T[0,:,:] is
  the covariance of the DC offset.

  Parameters:
  -----------
  T : 2D array with shape (p x n)
    DFT-transformed window with `p` channels and window length `n`.

  Returns
  -------
  out: a 3D tensor with shape (n x p x p)
    Stacked covariance matrices per frequency bin.

  See also
  --------
  band_cov : covariance matrix for a single frequency bin.

  Examples
  --------
  >>> p, n = 10, 128
  >>> X = np.random.randn(p, n)
  >>> T = cov_tens(np.fft.fft(X, axis=1))
  >>> Sig = np.sum(T[1:], 0)
  >>> C = np.cov(X, bias=True)
  >>> np.testing.assert_almost_equal(Sig, C)

  References
  ----------
  [1] Jason Farquhar. A linear feature space for simultaneous learning
  of spatio-spectral filters in BCI. Neural Networks, 22:1278--1285,
  2009.
  '''
  T = np.atleast_2d(T)
  C = np.asarray([band_cov(T[:, i]) for i in range(T.shape[1])])
  return C / T.shape[1] ** 2


def tile_tens3d(T, width=None):
  '''
  Tile the slabs of a 3D tensor T.

  Useful for visualization of stacks of arrays and 3D tensors.

  Parameters
  ----------
  T : 3D array
    The tensor to be tiled. The first dimension is rearranged, the 2D
    relation of the latter two are preserved.
  width : int, optional
    Determines how many 2D slabs are tiled horizontally. Tiling
    continues on the row below.

  Returns
  -------
  out : 2D array
    The tiled version of `T`. If `T` cannot completely cover `out`,
    the empty regions are filled with NaNs.

  Examples
  --------
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
  h = np.ceil(n / float(w))

  # construct 2D tile matrix
  R = np.vstack([T, np.nan * np.zeros((w * h - n, ch, cw))])
  R = R.reshape(h, w, ch, cw)
  R = np.concatenate(R, axis=1)
  R = np.concatenate(R, axis=1)
  return R
