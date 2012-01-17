# Very difficult to get right... but let's try again. We want to *generate* the
# sensor positions, and generate the correct labels.
# TODO:
# - use different spherical parameterization (with angle for front back, and
#   angle for left-right).
import numpy as np

def spher2cart(radius, inclination, azimuthal_angle):
  '''
  [1] http://en.wikipedia.org/wiki/File:Coord_system_SE_0.svg

  >>> spher2cart(2, 0, 0)
  (0., 0., 2.)

  >>> spher2cart(1, np.pi/2, 0)
  (0, 1, 0)

  >>> p = spher2cart(1, np.pi/2, np.pi/2)
  >>> np.round(p, decimals=8) #doctest: +NORMALIZE_WHITESPACE
  array([ 1.,  0.,  0.])
  '''
  x = radius * np.sin(inclination) * np.sin(azimuthal_angle)
  y = radius * np.sin(inclination) * np.cos(azimuthal_angle)
  z = radius * np.cos(inclination)
  return (x, y, z)

# Define landmarks on polar system:
# TODO
NASION = (1, (5./4) * np.pi/2., 0)
INION = (1, -(5./4) * np.pi/2., 0)

def per2spher(front_back, left_right):
  incl = np.interp(front_back, [0, 1], np.array([-1, 1]) * (5./4) * np.pi/2)
  azim = np.interp(left_right, [-.5, .5], np.array([-1, 1]) * (5./4) * np.pi/2)
  # FIXME: azim is wrong
  return (1, incl, azim)

print per2spher(0, 0), NASION
print per2spher(1, 0), INION

print spher2cart(*NASION)
print spher2cart(*INION)
