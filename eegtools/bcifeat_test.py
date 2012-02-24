# Copyright (c) 2011 Boris Reuderink
# License: BSD
import numpy as np
import bcifeat as bf


def test_detrend():
  x = np.linspace(0, 1, 1000)
  p0, p1, p2 = 20, 10 * x, 10 * x**2
  y = np.cos(x * 10 * 2 * np.pi)

  # test detrending precision
  np.testing.assert_almost_equal(bf.detrend(y), y, decimal=3)
  np.testing.assert_almost_equal(bf.detrend(y + p0), y, decimal=3)
  np.testing.assert_almost_equal(bf.detrend(y + p1), y, decimal=3)
  np.testing.assert_almost_equal(bf.detrend(y + p2, degree=2), y, decimal=1)

  # test default degree
  np.testing.assert_equal(bf.detrend(y + p2), bf.detrend(y + p2, degree=1))


def test_specweight():
  '''Test spectral band-pass weight for FFT filtering.'''
  def osc(f, fs):
    return np.sin(f * np.linspace(0, 2 * np.pi, fs))

  test_bands = [(2, 8), (8, 30), (30, 40)]
  test_rate = [128, 256]
  for rate in test_rate:
    for (low, high) in test_bands:
      print 'rate: %d, [%d, %d]' % (rate, low, high)
      weight = bf.specweight(rate, rate, [low, high])

      probes = [low - 1 , low + 1, np.mean([low, high]), high - 1, high + 1]
      resp = [np.linalg.norm(weight * np.fft.fft(osc(p, rate)))
        for p in probes]
      print np.asarray([probes, resp]).round(3)
      assert np.all(resp[0] < resp[1:-1]), 'low freqs not suppressed!'
      assert np.all(resp[-1] < resp[1:-1]), 'high freqs not suppressed!'
  

def test_bandcov():
  p, n = 10, 256
  tr_f = np.apply_along_axis(np.fft.rfft, 1, np.random.randn(p, n))
  for freq in range(1, tr_f.shape[1]):
    print 'freq:', freq
    # construct single-band data
    tr_b = np.where(np.atleast_2d(np.arange(tr_f.shape[1]) == freq), tr_f, 0)
    tr = np.apply_along_axis(np.fft.irfft, 1, tr_b)

    # calculate normal and DFT based covariance
    Cf = bf.bandcov(tr_f[:,freq])
    C = np.cov(tr, bias=True)

    # normalize
    k = 1. if freq == tr_f.shape[1] - 1 else 2.
    k /= n ** 2
    print 'k=%.2g' % k
    np.testing.assert_almost_equal(k * Cf, C)


def test_covtens():
  p, n = 12, 128
  X = np.random.randn(p, n)
  T = bf.covtens(np.fft.fft(X))
  np.testing.assert_almost_equal(np.sum(T[1:], 0), np.cov(X, bias=True))


def test_whitener():
  Sig = np.cov(np.random.rand(10, 100))
  W = bf.whitener(Sig)
  np.testing.assert_almost_equal(reduce(np.dot, [W.T, Sig, W]), np.eye(10))


def test_whitener_lowrank():
  '''Test whitener with low-rank covariance matrix'''
  Sig = np.eye(10)
  Sig[0, 0] = 0
  W = bf.whitener(Sig)
  np.testing.assert_almost_equal(reduce(np.dot, [W.T, Sig, W]), Sig)
