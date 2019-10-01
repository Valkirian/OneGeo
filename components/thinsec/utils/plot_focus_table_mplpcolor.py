import sys

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np


def plot_map(x_stride, y_stride, x_cols, y_rows, focus_map, resolution=1):

    x_lim = x_stride*x_cols
    y_lim = y_stride*y_rows
    xx, yy = np.meshgrid(np.arange(0, x_lim, resolution),
                         np.arange(0, y_lim, resolution))
    zz = focus_map(xx, yy)

    plt.pcolor(xx, yy, zz, cmap=cm.jet)
    ax = plt.gca()
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_xlim(0, x_lim)
    ax.set_ylim(0, y_lim)
    plt.colorbar()


if __name__ == "__main__":
    from plot_focus_table_common import get_map
    focus_map, _, _ = get_map(sys.argv[1])
    plot_map(18, 20, 9, 7, focus_map)
    plt.show()
