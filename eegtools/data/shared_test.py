import numpy as np
import shared 
def test_recording_construction():
  p, n = 32, 100
  X = np.zeros((p, n))
  chan_lab = ['chan%d' % c for c in range(p)]

  dt = np.ones(n-1) * .1
  dt[[0, n/2]] = np.nan  # insert discontinuities

  events = [[0, 1, 2, 0],                  # event type
           [10, 20, 20, 50],               # start
           [15, 20, 30, 100],              # end
           [np.nan, np.nan, 1e6, np.nan]]  # optional values

  event_lab = {0 : 'zero', 1 : 'one', 2 : 'two'}
  folds = [0, 0, 1, 1]

  r = shared.Recording(X=X, 
    chan_lab=chan_lab, 
    dt=dt, 
    events=events, 
    event_lab=event_lab, folds=folds, rec_id='test_rec', 
    license='test_license')

  # test storage
  np.testing.assert_equal(r.X, X)
  assert r.chan_lab == chan_lab
  np.testing.assert_equal(r.dt, dt)
  np.testing.assert_equal(r.events, events)
  assert r.event_lab == event_lab
  np.testing.assert_equal(r.folds, folds)
  assert r.rec_id == 'test_rec'
  assert r.license == 'test_license'

  # test derived properties
  np.testing.assert_equal(r.continuous_starts, [0, 1, n/2 + 1])
  assert r.sample_rate == 10.

  # test string representation
  assert str(r) == \
    'Recording "test_rec" (32 channels x 100 samples) at 10.00 Hz in ' \
    '3 continuous blocks, with 4 events in 3 classes.' \


def test_cache_path():
  import os
  import tempfile

  # create new cache path
  try:
    tmpdir = tempfile.mkdtemp()
    os.environ['EEGTOOLS_DATA_CACHE'] = tmpdir

    os.rmdir(tmpdir)  # make path non-existent
    assert shared.make_cache_path(shared.get_cache_path()) == tmpdir

    readme_fname = os.path.join(shared.get_cache_path(), 'README.txt')
    print readme_fname
    assert os.path.exists(readme_fname)
  finally:
    os.remove(os.path.join(tmpdir, 'README.txt')); 
    os.rmdir(tmpdir)

  # test default cache path
  del os.environ['EEGTOOLS_DATA_CACHE']
  assert shared.get_cache_path() == os.path.expanduser('~/eegtools_data')
