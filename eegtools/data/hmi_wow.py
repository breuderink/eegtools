#!/usr/bin/env python
import argparse, logging, collections
import numpy as np
from shared import Recording, data_source
import eegtools.io

'''
I just have to vent... what a mess:
- Logs and EEG recordings had different naming schemes, with exceptions,
- the log files have different fields for mouse and key events,
- the events in the log files are *almost* chronologically ordered,
- the time domains are non-linearly related,
- the markers indicating the correspondence between log and samples are
  ambiguous.
'''

# TODO:
# - Add subject list.

__all__ = ['load', 'subjects']
log = logging.getLogger(__name__)
LOGTIMERES = 1e-3

URL_TEMPLATE = 'https://s3-eu-west-1.amazonaws.com/bcidata.breuderink/' + \
  'hmi_awow/hmi-awow_%s.%s'

# These tuples link an text description to the code in the log file and a
# corresponding marker. It is like a Rosetta stone. Unfortunately, a single
# markers is sometimes used to indicate different events.
# This list was constructed by parsing the session log files.
LABEL_LOG_MARK = [
 ('mouse LMB pressed', 513, 1),
 ('mouse LMB released', 514, 2),
 ('mouse RMB pressed', 516, 4),
 ('mouse RMB released', 517, 5),
 ('mouse MMB pressed', 519, 7),
 ('mouse MMB released', 520, 8),  # Ambiguous marker!
 ('key pressed [Back]', 8, 8),    # Ambiguous marker!
 ('key pressed [Tab]', 9, 9),
 ('mouse scroll', 522, 10),
 ('key pressed [Return]', 13, 13),
 ('key pressed [Capital]', 20, 20),
 ('key pressed [Escape]', 27, 27),
 ('key pressed [Space]', 32, 32),
 ('key pressed [End]', 35, 35),
 ('key pressed [Home]', 36, 36),
 ('key pressed [Left]', 37, 37),
 ('key pressed [Up]', 38, 38),
 ('key pressed [Right]', 39, 39),
 ('key pressed [Down]', 40, 40),
 ('key pressed [Insert]', 45, 45),
 ('key pressed [Delete]', 46, 46),
 ('key pressed [0]', 48, 48),
 ('key pressed [1]', 49, 49),
 ('key pressed [2]', 50, 50),
 ('key pressed [3]', 51, 51),
 ('key pressed [4]', 52, 52),
 ('key pressed [5]', 53, 53),
 ('key pressed [6]', 54, 54),
 ('key pressed [7]', 55, 55),
 ('key pressed [8]', 56, 56),
 ('key pressed [9]', 57, 57),
 ('key pressed [A]', 65, 65),
 ('key pressed [B]', 66, 66),
 ('key pressed [C]', 67, 67),
 ('key pressed [D]', 68, 68),
 ('key pressed [E]', 69, 69),
 ('key pressed [F]', 70, 70),
 ('key pressed [G]', 71, 71),
 ('key pressed [H]', 72, 72),
 ('key pressed [I]', 73, 73),
 ('key pressed [J]', 74, 74),
 ('key pressed [K]', 75, 75),
 ('key pressed [L]', 76, 76),
 ('key pressed [M]', 77, 77),
 ('key pressed [N]', 78, 78),
 ('key pressed [O]', 79, 79),
 ('key pressed [P]', 80, 80),
 ('key pressed [Q]', 81, 81),
 ('key pressed [R]', 82, 82),
 ('key pressed [S]', 83, 83),
 ('key pressed [T]', 84, 84),
 ('key pressed [U]', 85, 85),
 ('key pressed [V]', 86, 86),
 ('key pressed [W]', 87, 87),
 ('key pressed [X]', 88, 88),
 ('key pressed [Y]', 89, 89),
 ('key pressed [Z]', 90, 90),
 ('key pressed [Lwin]', 91, 91),
 ('key pressed [F1]', 112, 112),
 ('key pressed [F2]', 113, 113),
 ('key pressed [F3]', 114, 114),
 ('key pressed [F4]', 115, 115),
 ('key pressed [F5]', 116, 116),
 ('key pressed [F6]', 117, 117),
 ('key pressed [F7]', 118, 118),
 ('key pressed [F8]', 119, 119),
 ('key pressed [F9]', 120, 120),
 ('key pressed [F10]', 121, 121),
 ('key pressed [F11]', 122, 122),
 ('key pressed [F12]', 123, 123),
 ('key pressed [Numlock]', 144, 144),
 ('key pressed [Lshift]', 160, 160),
 ('key pressed [Rshift]', 161, 161),
 ('key pressed [Lcontrol]', 162, 162),
 ('key pressed [Rcontrol]', 163, 163),
 ('key pressed [Lmenu]', 164, 164),
 ('key pressed [Rmenu]', 165, 165),
 ('key pressed [Browser_Back]', 166, 166),
 ('key pressed [Volume_Mute]', 173, 173),
 ('key pressed [Volume_Down]', 174, 174),
 ('key pressed [Volume_Up]', 175, 175),
 ('key pressed [Oem_Plus]', 187, 187),
 ('key pressed [Oem_Comma]', 188, 188),
 ('key pressed [Oem_Minus]', 189, 189),
 ('key pressed [Oem_Period]', 190, 190),
 ('key pressed [Oem_2]', 191, 191),
 ('key pressed [Oem_3]', 192, 192),
 ('key pressed [Oem_5]', 220, 220),
 ('key pressed [Oem_7]', 222, 222),
 ('key pressed [None]', 255, 255)]


def load(subject_id, ds=data_source()):
  # Load the continuous recording.
  edf = eegtools.io.load_edf(ds.open(URL_TEMPLATE % (subject_id, 'edf')))

  # Extract EEG data.
  eeg = slice(2, 16)
  chan_lab = edf.chan_lab[eeg]
  X = edf.X[eeg]

  # Kill ambiguous events.
  marker = edf.X[edf.chan_lab.index('MARKER')].astype(int)
  event_lab = dict([(m, l) for (l, i, m) in LABEL_LOG_MARK])

  meanings = collections.Counter([m for (_, _, m) in LABEL_LOG_MARK])
  for (m, count) in meanings.items():
    if count > 1:
      log.info('Purging ambiguous marker %d.' % m)
      marker[marker==m] = 0
      del event_lab[m]
  
  # Reorganize events.
  ei = np.flatnonzero(marker)
  E = np.vstack([marker[ei], ei, ei])

  return Recording(X=X, dt=np.diff(edf.time), chan_lab=chan_lab, events=E, 
    folds=np.ones(E.shape[1]), event_lab=event_lab, 
    rec_id='hmi-awow-%s' % subject_id)
