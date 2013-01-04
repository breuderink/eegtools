import logging, collections, operator, difflib
from scipy import optimize
import numpy as np
import hmi_wow


log = logging.getLogger(__name__)


def alignment(offsets1, offsets2, loss_fun=lambda d : d**2):
  '''
  Loss function for the alignment of events in two different streams. For each
  event, the nearest event in the second stream is found. The difference
  between the event and it's closest match is passed to a loss function.
  Note that this function is not symmetrical!

  Parameters
  ----------
  offsets1, offsets2: iterable
    Contain the offsets of comparable events in two different streams.
  loss_fun : function
    Function that accepts 

  Returns
  -------
  loss : Aggregated loss as computed by `loss_fun` over all events in
    `offsets1` matched to `offsets2`.

  '''
  # Sort one of the offsets for fast lookups.
  offsets2 = np.sort(offsets2)  

  loss = 0.
  for o in offsets1:
    n = neighbours(o, offsets2)
    if not n:
      log.debug('No neighbours found for offset %.4g.', o)
      continue

    # Compute minimal loss:
    diffs = offsets2[n] - o
    d = min(diffs, key=abs)
    log.debug('Differences are %s., minimum is %4g.', diffs, d)

    loss += loss_fun(d)

  return loss


def neighbours(t, sorted_offsets):
  # TODO: how to handle vector-valued ts?
  i = np.searchsorted(sorted_offsets, t)
  log.debug('i=%d', i)
  assert i == len(sorted_offsets) or t <= sorted_offsets[i]

  candidates = [i-1, i]
  neighbours = [i for i in candidates if 0 <= i < len(sorted_offsets)]
  log.debug('Found neighbours %s for offset %d.', neighbours, t)

  return neighbours


def test_alignment():
  assert alignment(range(3), range(3)) == 0
  assert alignment(range(10, 20), range(10, 20)) == 0
  assert alignment(range(10, 20), range(0, 30)) == 0
  assert alignment(range(3), range(3)[::-1]) == 0

  assert alignment(range(-1, 2), range(4)) == 1
  assert alignment(range(-1, 3), range(4)) == 1

  assert alignment([4], [-1, -2, -0]) == 4 ** 2
  assert alignment(range(3), []) == 0


def interval_check(off_a, off_b):
  '''
  Check for similarity of intervals invariant to linear transformations.
  This is performed by taking the difference between offsets (to make it
  invariant to translations), and subsequently calculating a ratio
  between subsequent differences. The latter is done through a
  logarithm.
  '''
  print np.diff(off_a)
  print np.diff(off_b)
  int_a = np.diff(np.log(np.diff(off_a)))
  int_b = np.diff(np.log(np.diff(off_b)))
  print np.vstack([int_a, int_b]).T
  np.testing.assert_allclose(int_a, int_b, atol=1e-5)


def align(events_a, off_a, events_b, off_b):
  '''
  Align two sparse series A and B. First, the longest matching subsequence is
  found. Then, a linear transformation is fitted to transform the matching
  events in A to the domain of B.
  @@FIXME
  '''
  off_a, off_b = np.atleast_1d(off_a, off_b)
  assert np.all(np.diff(off_a)) >= 0
  assert np.all(np.diff(off_b)) >= 0

  # First, try to find a matching subsequence. We assume that a
  # sufficient long subsequence does indeed exists.

  sm = difflib.SequenceMatcher(a=events_a, b=events_b)
  x, y = [], []
  for ai, bi, l in sm.get_matching_blocks():
    win_a, win_b = slice(ai, ai+l), slice(bi, bi+l)

    log.info('Found match: a[%d:%d] = b[%d:%d].', ai, ai+l, bi, bi+l)
    if l < 300:
      log.debug('Matching subsequence: %s', events_a[win_a])
    if l > 20:
      x.append(off_a[win_a])
      y.append(off_b[win_b])
   
  return np.hstack(x), np.hstack(y)


def transform(X_train, y_train, X_test):
  # Perform a sanity check
  # FIXME interval_check(off_a[win_a], off_b[win_b])
  
  # Now we have some matching events, we solve for a linear
  # transformation from A to B:
  X = np.vstack([off_a[win_a], np.ones(l)]).T
  y = off_b[win_b]
  w, res, _, _  = np.linalg.lstsq(X, y)

  log.info('MSE=%.3f', res / l)

  ws.append(w)
  return ws


def ridge(X, y, gamma):
  '''
  X w = y

  min ||Xw + y||^2 + ||Gw||^2

  w = (X^T X + G^T G)^{-1} X^T y
  '''
  X = np.atleast_2d(X)
  y = np.atleast_1d(y)
  n, p = X.shape
  
  A = np.linalg.inv(np.dot(X.T, X) + np.eye(p) * float(gamma) ** 2)
  w = reduce(np.dot, [A, X.T, y])
  return w


def phi(x, bases, sigma=1e2):
  '''Expansion to linear features and Gaussian functions centered at bases.'''
  x, bases = np.atleast_1d(x, bases)
  X = np.exp(-((x.reshape(-1, 1) - bases) / sigma) ** 2)
  X = np.vstack([x, np.ones(x.size), X.T]).T
  return X


if __name__ == '__main__':
  from hmi_wow import EVENTS
  import matplotlib.pyplot as plt; plt.ion()
  logging.basicConfig(level=logging.INFO)

  (E, edf, X, marker) = hmi_wow.load('1a-01')

  # Now we have a series of events form both the log and the EEG
  # recording. For mapping, there are a few problems: 
  # - the markers are not unique :/, 
  # - the  additional event information is stored in the log, so all
  # events need to be matched.
  #
  # First, we map the events from the log to corresponding markers,
  # and extract the markers form the continuous marker channel:
  S1 = np.asarray([(EVENTS[e][1], s) for (e, s) in E[:2].T])
  S2 = np.asarray([
    marker[np.flatnonzero(marker)], 
    np.flatnonzero(marker) / edf.sample_rate]).T
  


  bases = np.linspace(np.min(x), np.max(x), 20)
  phi = np.exp((-(x - bases.reshape(-1, 1)) ** 2) * 1e-5)
