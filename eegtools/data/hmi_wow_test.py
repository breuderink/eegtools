import numpy as np
import hmi_wow

from hmi_wow import EVENTS
if __name__ == '__main__':
  (E, edf, X, marker) = hmi_wow.load('1a-ch0')

  # Now we have a series of events form both the log and the EEG
  # recording. For mapping, there are a few problems: 
  # - the markers are not unique :/, 
  # - the  additional event information is stored in the log, so all
  # events need to be matched.
  #
  # First, we map the events from the log to corresponding markers,
  # and extract the markers form the continuous marker channel:
  S1 = np.asarray([EVENTS[e][1] for e in E[0]])
  S2 = marker[np.flatnonzero(marker)]

  # Our aim is now to find an alignment between the two sequences.
  # Simply said, for each element in S1 we want to find and element in
  # S2, such that:
  # - order is more or less preserved,
  # - the elements in S1 and S2 have the same value,
  # - memory use is not excessive (i.e. full Cartesian product of S1 and S2).

  # Strategy:
  # - look at string-alignment algorithms
  # - match with histogram?
  # - least square method to estimate offset(time) -> problem is
  # alignment again :/.

  # - find initial lag with brute-force method

  n = 500
  lag_s2 = np.argmax([np.sum(S1[:n]==np.roll(S2[:n], i)) for i in range(n)])


  # ---
  mark_event = sorted([(mark, code) for (code, (_, mark)) in EVENTS.items()])


