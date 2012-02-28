#!/usr/bin/env python
# -*- coding: utf-8
import os.path, urllib2, zipfile
from StringIO import StringIO
import numpy as np
from scipy import io
from shared import Recording, data_source

__all__ = ['load', 'subjects']

subjects = 'aa al av aw ay'.split()

URL_TR = 'http://bbci.de/competition/download/'\
  'competition_iii/berlin/100Hz/data_set_IVa_%s_mat.zip'
URL_TE = 'http://bbci.de/competition/iii/results/berlin_IVa/'\
  'true_labels_%s.mat'


LICENSE = '''Each participant has to agree to give reference to the group(s)
which recorded the data and to cite (one of) the paper listed in the respective
description in each of her/his publications where one of those data sets is
analyzed. Furthermore, we request each author to report any publication
involving BCI Competiton data sets to us for including it in our list.

[1] Guido Dornhege, Benjamin Blankertz, Gabriel Curio, and Klaus-Robert
    Müller. Boosting bit rates in non-invasive EEG single-trial
    classifications by feature combination and multi-class paradigms. IEEE
    Trans. Biomed. Eng., 51(6):993-1002, June 2004.

Note that the above reference describes an older experimental setup. A new
paper analyzing the data sets as provided in this competition and presenting
the feedback results will appear soon.
'''

def load_mat(mat_train, mat_test, rec_id):
  '''Load BCI Comp. 3.4a specific Matlab files.'''
  mat = io.loadmat(mat_train, struct_as_record=True)
  mat_true = io.loadmat(mat_test, struct_as_record=True)


  # get simple info from MATLAB files
  X, nfo, mrk = .1 * mat['cnt'].astype(float).T, mat['nfo'], mat['mrk']
  X, nfo, mrk = X.astype(np.float32), nfo[0][0], mrk[0][0]
  sample_rate = float((nfo['fs'])[0][0])
  dt = np.ones(X.shape[1]-1) / sample_rate
  chan_lab = [str(c[0]) for c in nfo['clab'].flatten()]

  # extract labels from both MATLAB files
  offy = mrk['pos'].flatten()
  tr_y = mrk['y'].flatten()
  all_y = mat_true['true_y'].flatten()
  assert np.all((tr_y == all_y)[np.isfinite(tr_y)]), 'labels do not match.'

  class_lab = [str(c[0]) for c in (mrk['className'])[0]]
  events = np.vstack([all_y, offy, offy + 3.5 * sample_rate]).astype(int)
  event_lab = dict(zip(np.unique(events[0]), class_lab))

  folds = np.where(np.isfinite(tr_y), -1, 1).tolist()

  return Recording(X=X, dt=dt, chan_lab=chan_lab, 
    events=events, event_lab=event_lab, folds=folds,
    rec_id=rec_id, license=LICENSE)


def load(subject, ds=data_source(), user='bci@mailinator.com', 
  password='laichucaij'):
  # get HTTP authentication going
  password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
  password_mgr.add_password(None, 'http://bbci.de', user, password)
  handler = urllib2.HTTPBasicAuthHandler(password_mgr)
  opener = urllib2.build_opener(urllib2.HTTPHandler, handler)
  urllib2.install_opener(opener)

  # Load the training set. We need to get a *seekable* file from a zip file.
  # Hence, StringIO is used.
  tr = zipfile.ZipFile(ds.open(URL_TR % subject))
  tr_mat = StringIO(tr.read('100Hz/data_set_IVa_%s.mat' % subject))  

  # Load test labels that were made available after the competition.
  te_mat = ds.open(URL_TE % subject)

  return load_mat(tr_mat, te_mat, 'bcicomp3.4a-%s' % subject)
