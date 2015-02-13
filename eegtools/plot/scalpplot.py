import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle

import positions

def plot_scalp(densities, sensors, sensor_locs=positions.POS_10_5, 
  plot_sensors=True, cmap=plt.cm.jet, clim=None):

  # add densities
  if clim == None:
    clim = [np.min(densities), np.max(densities)]
  locs = [positions.project_scalp(*sensor_locs[lab]) for lab in sensors]
  add_density(densities, locs, cmap=cmap, clim=clim)

  # setup plot
  MARGIN = 1.2
  plt.xlim(-MARGIN, MARGIN)
  plt.ylim(-MARGIN, MARGIN)
  plt.box(False)
  ax = plt.gca()
  ax.set_aspect(1.2)
  ax.yaxis.set_ticks([],[])
  ax.xaxis.set_ticks([],[])

  # add details
  add_head()
  if plot_sensors:
    add_sensors(sensors, locs)
 
def add_head():
  '''Draw head outline'''
  LINEWIDTH = 2
  nose = [(Path.MOVETO, (-.1, 1.)), (Path.LINETO, (0, 1.1)),
    (Path.LINETO, (.1, 1.))]

  lear = [(Path.MOVETO, (-1, .134)), (Path.LINETO, (-1.04, 0.08)),
    (Path.LINETO, (-1.08, -0.11)), (Path.LINETO, (-1.06, -0.16)),
    (Path.LINETO, (-1.02, -0.15)), (Path.LINETO, (-1, -0.12))]

  rear = [(c, (-px, py)) for (c, (px, py)) in lear]

  # plot outline
  ax = plt.gca()
  ax.add_artist(plt.Circle((0, 0), 1, fill=False, linewidth=LINEWIDTH))

  # add nose and ears
  for p in [nose, lear, rear]:
    code, verts = zip(*p)
    ax.add_patch(PathPatch(Path(verts, code), fill=False, linewidth=LINEWIDTH))


def add_sensors(labels, locs):
  '''Adds sensor names and markers'''
  for (label, (x, y)) in zip(labels, locs):
    if len(labels) < 32:
      plt.text(x, y + .03, label, fontsize=8, ha='center')
    plt.plot(x, y, 'k.', ms=2.)

def add_density(dens, locs, cmap=plt.cm.jet, clim=None):
  '''
  This function draws the densities using the locations provided in
  sensor_dict. The two are connected throught the list labels.  The densities
  are inter/extrapolated on a grid slightly bigger than the head using
  scipy.interpolate.rbf. The grid is drawn using the colors provided in cmap
  and clim inside a circle. Contours are drawn on top of this grid.
  '''
  RESOLUTION = 50
  RADIUS = 1.2
  xs, ys = zip(*locs)
  extent = [-1.2, 1.2, -1.2, 1.2]
  vmin, vmax = clim

  # interpolate
  # TODO: replace with Gaussian process interpolator. I don't trust SciPy's 
  # interpolation functions (they wiggle and they segfault).
  rbf = interpolate.Rbf(xs, ys, dens, function='linear')
  xg = np.linspace(extent[0], extent[1], RESOLUTION)
  yg = np.linspace(extent[2], extent[3], RESOLUTION)
  xg, yg = np.meshgrid(xg, yg)
  zg = rbf(xg, yg)

  # draw contour
  plt.contour(xg, yg, np.where(xg ** 2 + yg ** 2 <= RADIUS ** 2, zg, np.nan),
    np.linspace(vmin, vmax, 13), colors='k', extent=extent, linewidths=.3)

  # draw grid, needs te be last to enable plt.colormap() to work
  im = plt.imshow(zg, origin='lower', extent=extent, vmin=vmin, vmax=vmax, 
    cmap=cmap)

  # clip grid to circle
  patch = Circle((0, 0), radius=RADIUS, facecolor='none', edgecolor='none')
  plt.gca().add_patch(patch)
  im.set_clip_path(patch)
