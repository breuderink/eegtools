import numpy as np
import hmi_wow

def test_hmi_wow():
  r = hmi_wow.load('1a-01')
  assert str(r) == \
    'Recording "hmi-awow-1a-01" (14 channels x 596608 samples) ' + \
    'at 128.00 Hz in 1 continuous blocks, with 58836 events in 88 classes.'
