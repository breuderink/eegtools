#!/usr/bin/env python
import argparse, os.path
import numpy as np
from scipy import io
from shared import Recording, data_source

__all__ = ['load', 'subjects']

subjects = [i for i in range(12) if i not in [3, 8]]

URL_TEMPLATE = 'https://s3-eu-west-1.amazonaws.com/bcidata.breuderink/'\
  'reuderink_affpac/reuderink_affpac_s%d.mat.gz'

LICENSE = '''Please cite:

[1] Boris Reuderink, Mannes Poel, and Anton Nijholt. The impact of loss of
    control on movement BCIs. IEEE Transactions on Neural Systems and
    Rehabilitation Engineering, 19(6):628-637, December 2011. doi:
    10.1109/TNSRE.2011.2166562.

[2] Boris Reuderink, Anton Nijholt, and Mannes Poel. Affective Pacman: A
    frustrating game for brain-computer interface experiments. In Proceedings
    of the 3rd International Conference on Intelligent Technologies for Inter-
    active Entertainment (INTETAIN 2009), volume 9 of Lecture Notes of the
    Institute for Computer Sciences, Social Informatics and Telecommunications
    Engineering, pages 221-227. May 2009. doi: 10.1007/978-3-642-02315-6_23.
'''


EVENTS = [
  (1, 'index finger left hand key press'),
  (2, 'index finger right hand key press'),
  (3, 'index finger left hand key press error'),
  (4, 'index finger right hand key press error'),
  (5, 'visual screen freeze error'),
  #(10, 'init_level'),
  (11, 'next_level'),
  (12, 'pacman avatar died'),
  #(20, 'start_game'),
  #(21, 'end_game'),
  #(22, 'start_normal'),
  #(23, 'end_normal'),
  #(24, 'start_frustration'),
  #(25, 'end_frustration'),
  #(26, 'start_sam'),
  #(27, 'end_sam'),
  #(28, 'start_pause'),
  #(29, 'end_pause'),
  (90, 'keyboard error loss-of-control'),
  (100, 'valence'),
  (110, 'arousal'),
  (120, 'dominance'),
  ]

EVENT_OFFSETS = [
  (1, [0, 0]),
  (2, [0, 0]),
  (3, [0, 0]),
  (4, [0, 0]),
  (5, [0, 0]),
  (11, [0, 0]),
  (12, [0, 0]),
  ]

# Events with an extra value are stored separately:
COL_TO_EVENT = [(2, 90), (3, 100), (4, 110), (5, 120)]


def status_to_events(status, fs):
  events = []
  for marker, interval in EVENT_OFFSETS:
    offset = (fs * np.array(interval)).astype(int)
    intervals = np.flatnonzero(status==marker) + offset.reshape(-1, 1)
    events.append(
      np.vstack([marker * np.ones((1, intervals.shape[1]), int), intervals]))

  return np.hstack(events)


def block_to_events(I):
  events = []

  blocks = I[1]
  for bi in np.unique(blocks):
    if bi == -1:
      continue
    block_samp = np.flatnonzero(blocks == bi)
    start, end = np.min(block_samp), np.max(block_samp)

    for ci, ei in COL_TO_EVENT:
      events.append((ei, start, end, I[ci, block_samp[0]]))

  events = np.array(events, int).T

  # remove events marking the condition with full control
  events = events[:,~np.logical_and(events[0] == 90, events[3] == 0)]
  events[3, events[0] == 90] = 0
  return events



def load(subject_id, ds=data_source()):
  matfile = ds.open(URL_TEMPLATE % subject_id)
  mat = io.loadmat(matfile, struct_as_record=True)

  X = mat['X'].astype(np.float32)
  dt = np.diff(mat['I'][0])
  chan_lab = [str(l[0]) for l in mat['chann'].flat]

  # create event matrix
  sample_rate = 1./np.median(dt)
  status_events = status_to_events(mat['Y'], sample_rate)
  block_events = block_to_events(mat['I'])
  events = np.hstack([
    np.vstack([status_events, np.zeros((1, status_events.shape[1]), int)]), 
    block_events
    ])
  events = events[:,np.argsort(events[1])]  # sort events on start time

  # fill other attributes
  event_lab = dict(EVENTS)
  folds = mat['I'][1][events[1]].astype(int)  # use block number as fold id

  # construct final record
  return Recording(X=X, dt=dt, chan_lab=chan_lab, events=events, 
    folds=folds, event_lab=event_lab, 
    rec_id='reuderink-affpac-s%d' % subject_id, license=LICENSE)
