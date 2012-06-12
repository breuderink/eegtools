import numpy as np
from scipy import signal
from sklearn import svm, pipeline, base, metrics
import eegtools

'''
Example of a traditional, but simplified pipeline to classify power changes in
the EEG caused by imagery of movements.
'''

# download and load dataset from BCI competition 3
d = eegtools.data.bcicomp3_4a.load('aa')
print d

# create band-pass filter for the  8--30 Hz where the power change is expected
(b, a) = signal.butter(3, np.array([8, 30]) / (d.sample_rate / 2), 'band')

# band-pass filter the EEG
d.X = signal.lfilter(b, a, d.X, 1)

# extract trials
offsets = [0.5 * d.sample_rate, np.median(d.events[2] - d.events[1])]
print 'Extracting interval of %s samples from trial start.' % offsets
trials, _ = eegtools.featex.windows(d.events[1], offsets, d.X)

# extract labels and folds
y, folds = d.events[0], d.folds


# create sklearn-compatible feature extraction
class CSP_feat(base.BaseEstimator, base.TransformerMixin):
  def fit(self, X, y):
    class_covs = []

    # calculate per-class covariance
    for ci in np.unique(y): 
      class_covs.append(np.cov(np.hstack(X[y==ci])))

    assert len(class_covs) == 2

    # calculate CSP projection
    self.W = eegtools.spatfilt.csp(class_covs[0], class_covs[1], 6)
    return self


  def transform(self, X):
    # Note that the projection on the spatial filter expects zero-mean data.
    return np.asarray([np.var(np.dot(self.W, trial), axis=1) for trial in X])


pipe = pipeline.Pipeline([
  ('csp feature extraction', CSP_feat()),
  ('svm', svm.SVC(kernel='linear')),
  ])

# separate mask for train and test data
test = folds == 1
train = ~test

# train model
pipe.fit(trials[train], y[train])

# make predictions on unseen test data
y_true = y[test]
y_pred = pipe.predict(trials[test])

# show results
print metrics.classification_report(y_true, y_pred)
