import numpy as np
from scipy import signal
from sklearn import svm, pipeline, base, metrics
import eegtools

'''
This example demonstrates the use of the common spatial patterns (CSP)
algorithm [1] on EEG data with motor imagery form BCI competition 3.4a.
For classification, scikit-learn (http://scikit-learn.org) is used.

[1] Zoltan J. Koles. The quantitative extraction and topographic mapping
of the abnormal components in the clinical EEG. Electroencephalography
and Clinical Neurophysiology, 79(6):440--447, December 1991.
'''

# Automatically download and load dataset from BCI competition 3. For
# experimental info see: http://www.bbci.de/competition/iii/#data_set_iva .
d = eegtools.data.bcicomp3_4a.load('aa')
print d

# create band-pass filter for the  8--30 Hz where the power change is expected
(b, a) = signal.butter(3, np.array([8, 30]) / (d.sample_rate / 2), 'band')

# band-pass filter the EEG
d.X = signal.lfilter(b, a, d.X, 1)

# extract trials
offsets = np.array([0.5 * d.sample_rate, np.median(d.events[2] - d.events[1])])
print 'Extracting interval of %s sec from trial start.' % \
  (offsets / d.sample_rate)
trials, _ = eegtools.featex.windows(d.events[1], offsets, d.X)

# extract labels and folds
y, folds = d.events[0], d.folds


# Create sklearn-compatible feature extraction and classification pipeline:
class CSP(base.BaseEstimator, base.TransformerMixin):
  def fit(self, X, y):
    class_covs = []

    # calculate per-class covariance
    for ci in np.unique(y): 
      class_covs.append(np.cov(np.hstack(X[y==ci])))
    assert len(class_covs) == 2

    # calculate CSP spatial filters
    self.W = eegtools.spatfilt.csp(class_covs[0], class_covs[1], 6)
    return self


  def transform(self, X):
    # Note that the projection on the spatial filter expects zero-mean data.
    return np.asarray([np.dot(self.W, trial) for trial in X])


class ChanVar(base.BaseEstimator, base.TransformerMixin):
  def fit(self, X, y): return self
  def transform(self, X):
    return np.var(X, axis=2)  # X.shape = (trials, channels, time)


pipe = pipeline.Pipeline([
  ('csp', CSP()),
  ('chan_var', ChanVar()),
  ('svm', svm.SVC(kernel='linear')),
  ])

# Create mask for same train and test data as used in competition:
test = folds == 1
train = ~test

# train model
pipe.fit(trials[train], y[train])

# make predictions on unseen test data
y_true = y[test]
y_pred = pipe.predict(trials[test])

# Show results. Competition results are available on
# http://www.bbci.de/competition/iii/results/index.html#berlin1
print metrics.classification_report(y_true, y_pred)
