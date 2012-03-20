import numpy as np
import spatfilt as sf

def test_whitener():
  Sig = np.cov(np.random.rand(10, 100))
  W = sf.whitener(Sig)
  np.testing.assert_almost_equal(reduce(np.dot, [W.T, Sig, W]), np.eye(10))


def test_whitener_lowrank():
  '''Test whitener with low-rank covariance matrix'''
  Sig = np.eye(10)
  Sig[0, 0] = 0
  W = sf.whitener(Sig)
  np.testing.assert_almost_equal(reduce(np.dot, [W.T, Sig, W]), Sig)


def test_select_channels():
  X = np.random.rand(10, 100)
  for keep in [[0, 1, -3, 2], (np.arange(10) % 2 == 0).astype(bool)]:
    W = sf.select_channels(X.shape[0], keep)
    print W.shape
    np.testing.assert_equal(np.dot(W, X), X[keep])


def test_car():
  X = np.random.rand(10, 4)
  W = sf.car(X.shape[1])
  np.testing.assert_almost_equal(
    np.dot(X, W), X - np.mean(X, axis=1).reshape(-1, 1))


def test_csp_base():
  p, n = 8, 100

  X_a = np.random.randn(p, n)
  X_b = np.random.randn(p, n)

  C_a = np.cov(X_a)
  C_b = np.cov(X_b)

  W = sf.csp_base(C_a, C_b)
  assert W.shape == (p, p)

  # Use W X X^T W^T ~ W C W^T to extract projected (co)variance:
  D_a = reduce(np.dot, [W, C_a, W.T])
  D_b = reduce(np.dot, [W, C_b, W.T])

  # W C W^T = I:
  np.testing.assert_almost_equal(D_a + D_b, np.eye(W.shape[1]), 
    err_msg='Joint covariance is not the identity matrix.')

  # W C_a W^T = D:
  np.testing.assert_almost_equal(np.diag(np.diag(D_a)), D_a,
    err_msg='Class covariance is not diagonal.')

  # D_ii < D_(i+1, i+1) with D = D_a:
  assert np.all(np.diff(np.diag(D_a)) >= 0), \
    'Class variance is not ascending.'


def test_outer_n():
  np.testing.assert_equal(sf.outer_n(1), [0])
  np.testing.assert_equal(sf.outer_n(2), [0, -1])
  np.testing.assert_equal(sf.outer_n(3), [0, 1, -1])
  np.testing.assert_equal(sf.outer_n(6), [0, 1, 2, -3, -2, -1])


def test_csp():
  p, n = 8, 100

  X_a = np.random.randn(p, n)
  X_b = np.random.randn(p, n)

  C_a = np.cov(X_a)
  C_b = np.cov(X_b)

  W_full = sf.csp_base(C_a, C_b)
  W = sf.csp(C_a, C_b, 7)

  np.testing.assert_equal(W, W_full[[0, 1, 2, 3, -3, -2, -1]])
