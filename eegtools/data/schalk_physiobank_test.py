import schalk_physiobank

def test_s1():
  r = schalk_physiobank.load(1)
  assert str(r) == 'Recording "schalk-physiobank-s1" (64 channels x 259520 '\
    'samples) at 160.00 Hz in 14 continuous blocks, with 362 events in 11 '\
    'classes.'

def test_s14():
  '''Test subject 14 which contains runs *without* any event.'''
  r = schalk_physiobank.load(14)
  assert str(r) == 'Recording "schalk-physiobank-s14" (64 channels x 255520 '\
    'samples) at 160.00 Hz in 14 continuous blocks, with 361 events in 11 '\
    'classes.'
