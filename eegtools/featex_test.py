# Copyright (c) 2011 Boris Reuderink
# License: BSD
import numpy as np
from scipy import signal
import featex as fe


def test_spec():
  def spec(x):
    return np.fft.rfft(signal.detrend(x) * np.hanning(x.size))

  T = np.random.randn(10, 20, 30) + 40
  for ax in range(3):
    S = fe.spec(T, axis=ax)
    np.testing.assert_almost_equal(S, np.apply_along_axis(spec, ax, T))


def test_spec_weight():
  '''Test spectral band-pass weight for FFT filtering.'''
  def osc(f, fs):
    return np.sin(f * np.linspace(0, 2 * np.pi, fs))

  test_bands = [(2, 8), (8, 30), (30, 40)]
  test_rate = [128, 256]
  for rate in test_rate:
    for (low, high) in test_bands:
      print 'rate: %d, [%d, %d]' % (rate, low, high)
      weight = fe.spec_weight(rate, rate, [low, high])

      probes = [low - 1 , low + 1, np.mean([low, high]), high - 1, high + 1]
      resp = [np.linalg.norm(weight * np.fft.fft(osc(p, rate)))
        for p in probes]
      print np.asarray([probes, resp]).round(3)
      assert np.all(resp[0] < resp[1:-1]), 'low freqs not suppressed!'
      assert np.all(resp[-1] < resp[1:-1]), 'high freqs not suppressed!'
  

def test_band_cov():
  p, n = 10, 256
  tr_f = np.apply_along_axis(np.fft.rfft, 1, np.random.randn(p, n))
  for freq in range(1, tr_f.shape[1]):
    print 'freq:', freq
    # construct single-band data
    tr_b = np.where(np.atleast_2d(np.arange(tr_f.shape[1]) == freq), tr_f, 0)
    tr = np.apply_along_axis(np.fft.irfft, 1, tr_b)

    # calculate normal and DFT based covariance
    Cf = fe.band_cov(tr_f[:,freq])
    C = np.cov(tr, bias=True)

    # normalize
    k = 1. if freq == tr_f.shape[1] - 1 else 2.
    k /= n ** 2
    print 'k=%.2g' % k
    np.testing.assert_almost_equal(k * Cf, C)


def test_cov_tens():
  p, n = 12, 128
  X = np.random.randn(p, n)
  T = fe.cov_tens(np.fft.fft(X))
  np.testing.assert_almost_equal(np.sum(T[1:], 0), np.cov(X, bias=True))


def test_whitener():
  Sig = np.cov(np.random.rand(10, 100))
  W = fe.whitener(Sig)
  np.testing.assert_almost_equal(reduce(np.dot, [W.T, Sig, W]), np.eye(10))


def test_whitener_lowrank():
  '''Test whitener with low-rank covariance matrix'''
  Sig = np.eye(10)
  Sig[0, 0] = 0
  W = fe.whitener(Sig)
  np.testing.assert_almost_equal(reduce(np.dot, [W.T, Sig, W]), Sig)
