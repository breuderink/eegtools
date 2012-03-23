# Copyright (c) 2011 Boris Reuderink
# License: BSD
import numpy as np
from scipy import signal
import featex as fe


def test_window():
  X = np.random.randn(3, 50)
  W, ii = fe.windows([0, 10, 12, 49], [-2, 3], X)
  print W, ii

  np.testing.assert_almost_equal(ii, [10, 12])
  assert(W.shape[0] == 2)
  np.testing.assert_equal(W[0], X[:,8:13])
  np.testing.assert_equal(W[1], X[:,10:15])


def test_spec():
  def spec_1d(x):
    return np.fft.rfft(signal.detrend(x) * np.hanning(x.size))

  T = np.random.randn(10, 20, 30) + 40
  for ax in range(3):
    S = fe.spec(T, axis=ax)
    np.testing.assert_almost_equal(S, np.apply_along_axis(spec_1d, ax, T))


def test_spec_weight():
  freqs = np.fft.fftfreq(128, d=1./128)
  for bleed in [3, 7, 15, 30]:
    print 'bleed = %d.' % bleed
    lp = fe.spec_weight(freqs, lp=30., bleed=bleed)
    hp = fe.spec_weight(freqs, hp=7., bleed=bleed)
    bp = fe.spec_weight(freqs, lp=30., hp=7, bleed=bleed)

    print np.vstack([freqs, lp, hp, bp]).T

    # test lp
    np.testing.assert_almost_equal(lp[np.abs(freqs)<= 30-bleed/2], 1)
    np.testing.assert_almost_equal(lp[np.abs(freqs)>= 30+bleed/2], 0)
    
    # test hp
    np.testing.assert_almost_equal(hp[np.abs(freqs)<= 7-bleed/2], 0)
    np.testing.assert_almost_equal(hp[np.abs(freqs)>= 7+bleed/2], 1)

    # test bp
    np.testing.assert_almost_equal(bp, np.min([hp, bp], axis=0))


def test_band_cov():
  p, n = 10, 256
  win_f = np.apply_along_axis(np.fft.rfft, 1, np.random.randn(p, n))
  for freq in range(1, win_f.shape[1]):
    print 'freq:', freq
    # construct single-band data
    tr_b = np.where(np.atleast_2d(np.arange(win_f.shape[1]) == freq), win_f, 0)
    tr = np.apply_along_axis(np.fft.irfft, 1, tr_b)

    # calculate normal and DFT based covariance
    Cf = fe.band_cov(win_f[:,freq])
    C = np.cov(tr, bias=True)

    # normalize
    k = 1. if freq == win_f.shape[1] - 1 else 2.
    k /= n ** 2
    print 'k=%.2g' % k
    np.testing.assert_almost_equal(k * Cf, C)


def test_cov_tens():
  p, n = 12, 128
  X = np.random.randn(p, n)
  T = fe.cov_tens(np.fft.fft(X))
  np.testing.assert_almost_equal(np.sum(T[1:], 0), np.cov(X, bias=True))


