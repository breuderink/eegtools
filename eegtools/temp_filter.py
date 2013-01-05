import logging
import numpy as np

log = logging.getLogger(__name__)

def zero_pad(x, n):
  x = np.atleast_1d(x)
  assert x.size <= n
  return np.hstack([x, np.zeros(n - x.size)])


def fast_convolve(x, h, step=16):
  fft, ifft = np.fft.rfft, np.fft.irfft
  x, h = np.atleast_1d(x, h)
  y = []

  # use variables similar to borgerding2006tos:
  l = step
  p = h.size
  n = step + p - 1  # size of FFT

  H = fft(zero_pad(h, n))  # convert filter to frequency domain
  log.info('H.shape = %s', H.shape)

  for i in range(0, x.size - n + 1, step):
    win = slice(i, i+n)
    log.info('Processing %s...', win)
    
    x_frag = zero_pad(x[win], n)
    y_frag = ifft(fft(x_frag) * H)  # filter in frequency domain
    y.append(y_frag[-l:])
    
  return np.hstack(y)


def test_fast_convolve():
  #x = np.random.randn(256)
  x = np.zeros(1024)
  x[[10, 100]] = 1
  fil = np.hanning(32)

  ref = np.convolve(x, fil, 'valid')
  y = fast_convolve(x, fil)

  import matplotlib.pyplot as plt;
  plt.plot(x, label='x', lw=1)
  plt.plot(ref, label='good', lw=2)
  plt.plot(y, lw=2, label='fft-variant')
  plt.legend()
  plt.show()

  np.testing.assert_equal(ref, y)
