'''
Copyright (c) 2014 Boris Reuderink.
'''
import logging
from bdf import *

log = logging.getLogger(__name__)

if __name__ == '__main__':
  fname = 'test.bdf'
  d = load_bdf('test.bdf')

  import pylab as plt; plt.ion()

  plt.plot(d.X[:,:1000].T + 200 * np.arange(d.X.shape[0]), c='k')


  #with open(fname, 'rb') as f:
  #  reader = BaseBDFReader(f)
  #  reader.read_header()
  #  h = reader.header
  #  print h
