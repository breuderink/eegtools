#!/usr/bin/env python
import operator, re, logging, os.path
import numpy as np
import eegtools.io
from shared import Recording, data_source

__all__ = ['load', 'subjects']

subjects = range(1, 110)

LICENSE = '''This dataset was created and contributed to PhysioNet by
the developers of the BCI2000 instrumentation system, which they used
in making these recordings. The system is described in:

[1] Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N.,
    Wolpaw, J.R. BCI2000: A General-Purpose Brain-Computer Interface
    (BCI) System. IEEE Transactions on Biomedical Engineering
    51(6):1034-1043, 2004. [In 2008, this paper received the Best
    Paper Award from IEEE TBME.]

Please cite this publication and www.bci2000.org when referencing this
material, and also include the standard citation for PhysioNet:

[2] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG,
    Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank,
    PhysioToolkit, and PhysioNet: Components of a New Research
    Resource for Complex Physiologic Signals. Circulation
    101(23):e215-e220 [Circulation Electronic Pages;
    http://circ.ahajournals.org/cgi/content/full/101/23/e215]; 2000
    (June 13).
'''

URL_TEMPLATE = 'http://www.physionet.org/pn4/eegmmidb/'\
  'S%(subject)03d/S%(subject)03dR%(run)02d.edf'

EVENTS = [
  (0, 'relax'),
  (1, 'baseline eyes open'),
  (2, 'baseline eyes closed'),
  (3, 'move left hand'),
  (4, 'move right hand'),
  (5, 'imagine move left hand'), 
  (6, 'imagine move right hand'),
  (7, 'move left right hand'), 
  (8, 'move feet'),
  (9, 'imagine move left right hand'), 
  (10, 'imagine move feet'),
  ]

# Translate EDF+ events to our events based on task
B1 = dict(T0=1)
B2 = dict(T0=2)
T1 = dict(T0=0, T1=3, T2=4)
T2 = dict(T0=0, T1=5, T2=6)
T3 = dict(T0=0, T1=7, T2=8)
T4 = dict(T0=0, T1=9, T2=10)
TASKS = [B1, B2] + [T1, T2, T3, T4] * 3

# Clean labels to lookup
# TODO: replace with labels in EEGtools when ready.
KNOWN_LABS = 'FC5 FC3 FC1 FCz FC2 FC4 FC6 C5 C3 C1 Cz C2 C4 C6 CP5 CP3 CP1 CPz\
  CP2 CP4 CP6 Fp1 Fpz Fp2 AF7 AF3 AFz AF4 AF8 F7 F5 F3 F1 Fz F2 F4 F6 F8 FT7\
  FT8 T7 T8 T9 T10 TP7 TP8 P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz PO4 PO8 O1\
  Oz O2 Iz'.split()

log = logging.getLogger(__name__)


def gen_urls(subject, url_template=URL_TEMPLATE):
  '''Return EDF+ URLs for a given subject and template'''
  runs = np.arange(14) + 1
  return [url_template % dict(subject=subject, run=run) 
    for run in runs], runs


def load_schalk_run(edf, run):
  '''Read an EDF+ file corresponding to a run, and return a proper dataset.'''
  task_dic = TASKS[run - 1]
  log.debug('Processing run %d, with dictionary %s' % (run, task_dic))
  d = eegtools.io.load_edf(edf)

  fs, events = d.sample_rate, []
  for (start, duration, texts) in  d.annotations:
    for label in texts:
      events.append(
        (task_dic[str(label)], start * fs, (start + duration) * fs))
  return d, np.asarray(events, int).T.reshape(3, -1)


def clean_chan_lab(feat_lab):
  '''Replace channel labels with matching, correctly capitalized labels.'''
  lookup = dict((l.lower(), l) for l in KNOWN_LABS)
  return [lookup[l.strip('.').lower()] for l in feat_lab]


def concatenate_events(events, block_lens):
  '''Concatenate events indexing different runs.'''
  def shift_events(events, offset):
      events = events.copy()
      events[1:] += offset
      return events

  offset = np.cumsum([0] + block_lens[:-1])
  return np.hstack([shift_events(be, o) for (be, o) in zip(events, offset)])


def block_dt(n, sample_rate):
  '''Generate dt array for a given length and sample_rate.'''
  dt = np.ones(n) * 1./sample_rate
  dt[0] = np.nan
  return dt


def load(subject, ds=data_source(), url_template=URL_TEMPLATE):
  '''Each subject performed 14 experimental runs: two one-minute
  baseline runs (one with eyes open, one with eyes closed), and three
  two-minute runs of each of the four following tasks:

  1) A target appears on either the left or the right side of the
  screen. The subject OPENS AND CLOSES THE CORRESPONDING FIST until
  the target disappears. Then the subject relaxes.

  2) A target appears on either the left or the right side of the
  screen. The subject IMAGINES OPENING AND CLOSING THE CORRESPONDING
  FIST until the target disappears. Then the subject relaxes.

  3) A target appears on either the top or the bottom of the screen.
  The subject OPENS AND CLOSES EITHER BOTH FISTS (if the target is on
  top) or BOTH FEET (if the target is on the bottom) until the target
  disappears. Then the subject relaxes.

  4) A target appears on either the top or the bottom of the screen.
  The subject IMAGINES OPENING AND CLOSING EITHER BOTH FISTS (if the
  target is on top) OR BOTH FEET (if the target is on the bottom)
  until the target disappears. Then the subject relaxes.

  The EDF+ file contains the duration of each event, however it seems
  to be slightly variable. Based on the first subjects, it seems that
  between each task there is a resting task, and both the active task
  and resting task take about 4.15 +- .1 seconds.

  The file headers for one subject all contain the same start
  timestamp, so we cannot estimate the time between runs. 
  '''
  urls, runs = gen_urls(subject, url_template)
  log.debug('Generated URLs: %s.' % (urls,))
  
  # Load runs for this subject:
  rec = [load_schalk_run(ds.open(u), r) for (u, r) in zip(urls, runs)]
  runs, events = zip(*rec)

  # Combine information from different runs:
  X = np.hstack([r.X.astype(np.float32) for r in runs])
  dt = np.hstack([block_dt(r.X.shape[1], r.sample_rate) for r in runs])[1:]
  chan_lab = clean_chan_lab(runs[0].chan_lab)
  folds = np.hstack([
    np.ones(e.shape[1], int) * i for (i, e) in enumerate(events)])
  E = concatenate_events(events, [r.X.shape[1] for r in runs])


  return Recording(X=X, dt=dt, chan_lab=chan_lab, events=E, folds=folds, 
    event_lab=EVENTS, rec_id='schalk-physiobank-s%d' % subject, 
    license=LICENSE)
