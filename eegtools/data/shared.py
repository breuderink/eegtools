import os
import textwrap
import warnings
import numpy as np

'''
EEGdata provides easy access to neurophysiological recordings, with
brain-computer interface (BCI) research in mind. In short, a BCI is
like speech recognition, but for brain signals.

The aim is to make publicly available datasets available to
researchers by automatically downloading and parsing them, and provide
the data in a no-frills format.
'''


CACHE_VAR = 'EEGTOOLS_DATA_CACHE'
CACHE_PATH = '~/eegtools_data'


class Recording:
  '''
  Recording is the format in which eegtools.data delivers the imported
  datasets. It's main ingredient is the matrix X that holds the
  recording, and an event matrix that annotates the samples in X.
  '''
  def __init__(self, X=None, dt=None, chan_lab=[], events=[], folds=None,
    event_lab=[], rec_id='', license=''):
    '''
    Initializes the recording.

    Parameters
    ----------
    X : array of shape (p, n)
      Contains the n samples for p sensors.

    dt : array of shape (n - 1)
      Describes the difference in time between consecutive samples of
      X. Both continuous and interrupted sessions can be stored this
      way. If the temporal difference is unknown, NaN can be used to
      indicate this difference.

    chan_lab : list with p strings
      Contains a label for each sensor. For EEG sensors, preferably
      labels from the (extended) 10-20 system are used.

    events : array of shape (>= 3, m)
      Describes m events. The rows of events contain respectively the
      event id, the start and the end of event as zero-based indices
      matching X. An optional extra rows can be used for additional
      values (e.g. for user reported emotional ratings, pressure on a
      switch etc.)

    folds : array of length m
      Contains a value for each event, describing a grouping of
      events. These groups can be used for cross-validation.

    event_lab : dictionary
      Is a lookup table that assigns a textual meaning to event ids.
      The values of the dictionary describe tag-like properties of the
      event.

    rec_id : string
      Is a string that identifies both the experiment and the subject.

    license: string
      The license under which the data is made available.
    '''
    self.X = np.atleast_2d(X).astype(np.float32)
    self.dt = dt = np.atleast_1d(dt)
    self.chan_lab = chan_lab
    self.events = events = np.atleast_2d(events)
    self.folds = folds = np.atleast_1d(folds).astype(int)
    self.event_lab = event_lab = dict(event_lab)
    self.rec_id = rec_id
    self.license = license

    if not self.rec_id:
      warnings.warn('No identifier (rec_id) provided for recording.')
    if not self.license:
      warnings.warn('No license provided for recording.')

    assert self.X.ndim == 2  # sensors x time
    p, n = self.X.shape
    assert len(self.chan_lab) == p
    assert self.dt.size == n - 1, \
      'Temporal difference dt has shape %s, should be %s' % \
        (self.dt.shape, (n - 1,))
    
    # check for matching events
    event_ids = set(np.unique(self.events[0]))
    event_lab_ids = set(self.event_lab.keys())
    assert len(event_ids.difference(event_lab_ids)) == 0, \
      'Unique events %s do not match events in event_lab %s!' % \
        (list(event_ids), list(event_lab_ids))

    # check event conventions
    ids, starts, ends = events[0], events[1], events[2]
    duration = ends - starts
    assert np.all(duration >= 0), \
      'Events should have a non-negative duration.'
    assert np.all(starts >= 0), \
      'Event starts before data stream X starts.'
    assert np.all(ends <= X.shape[1]), \
      'Events continue after data stream X ended.'
    assert np.all(np.diff(starts) >= 0), \
      'The starts of the events should be sorted chronologically.'

    # check folds
    assert folds.size == events.shape[1], 'Expected a fold number per event.'


  @property
  def sample_rate(self):
    '''Estimate sample rate based on dt.'''
    dt = self.dt 
    return 1./np.median(dt[np.isfinite(dt)])


  @property
  def continuous_starts(self):
    '''Return indices of starts of new continuous blocks'''
    return np.hstack([[0], 1 + np.flatnonzero(self.dt != 1./self.sample_rate)])

  
  def __str__(self):
    return ('Recording "%(rid)s" (%(p)d channels x %(n)d samples) '
      'at %(fs).2f Hz in %(blocks)d continuous blocks, '
      'with %(nevents)d events in %(nevent_types)d classes.') % \
      dict(p=self.X.shape[0], n=self.X.shape[1], fs=self.sample_rate,
        nevents=self.events.shape[1], nevent_types=len(self.event_lab), 
        blocks=self.continuous_starts.size, rid=self.rec_id)


def print_story(events, sample_rate):
  '''Not fully complete helper function to describe the story of the events.
  '''
  dic = dict(EVENTS)
  for (ei, start, end, optional) in events.T:
    print '%s @ %.2fs (%.2fs long) -> %d.' % (
      dic[ei].ljust(30), 
      start / sample_rate, 
      (end - start) / sample_rate, 
      optional
      )


def data_source():
  return np.DataSource(make_cache_path(get_cache_path()))


def get_cache_path():
  '''Get path for caching downloaded files. The location is indicated
  by the EEGTOOLS_DATA_CACHE environment variable. When the path does not
  exist, it is created and a README.txt with instructions is written
  to this location.

  Returns a string containing the path.
  '''
  return os.environ.get(CACHE_VAR, os.path.expanduser(CACHE_PATH))


def make_cache_path(path):
  '''Create path for caching downloaded files. When the path does not
  exist it is created, and a README.txt with instructions is written
  to this location.

  Returns a string containing the path.
  '''
  if not os.path.exists(path):
    os.makedirs(path)

  readme = os.path.join(path, 'README.txt')
  if not os.path.exists(readme):
    with open(readme, 'w') as f:
      f.write(textwrap.dedent(
        '''\
        This directory was created by eegtools to cache downloaded BCI
        datasets. It is safe to remove the cached files in this
        directory, but doing so will result in a performance penalty.

        To change this path, set the %(env)s environment variable with
        the path of your preference. For example:
        
            $ export %(env)s=%(cache_path)s

        That is all.
        ''' % dict(env=CACHE_VAR, cache_path=CACHE_PATH)))

  return path
