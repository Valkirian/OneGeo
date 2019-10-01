import sys

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
import numpy as np

from plot_focus_table_common import get_map

focus_map = get_map(sys.argv[1])

resolution = 1
xx = np.arange(0, 9*18, resolution)
xlen = len(xx)
yy = np.arange(0, 7*20, resolution)
ylen = len(yy)
xx, yy = np.meshgrid(xx, yy)
zz = focus_map(xx, yy)

fig = plt.figure()
ax = fig.gca(projection='3d')

colortuple = ('y', 'b')
colors = np.empty(xx.shape, dtype=str)
for y in range(ylen):
    for x in range(xlen):
        colors[y, x] = colortuple[(x + y) % len(colortuple)]

surf = ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, facecolors=colors,
        linewidth=0, antialiased=False)

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')

plt.show()
