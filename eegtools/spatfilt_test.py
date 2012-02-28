import numpy as np
import spatfilt

def test_whitener():
  Sig = np.cov(np.random.rand(10, 100))
  W = spatfilt.whitener(Sig)
  np.testing.assert_almost_equal(reduce(np.dot, [W.T, Sig, W]), np.eye(10))


def test_whitener_lowrank():
  '''Test whitener with low-rank covariance matrix'''
  Sig = np.eye(10)
  Sig[0, 0] = 0
  W = spatfilt.whitener(Sig)
  np.testing.assert_almost_equal(reduce(np.dot, [W.T, Sig, W]), Sig)
