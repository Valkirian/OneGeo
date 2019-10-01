import os
import sys

import cv2
from matplotlib import pyplot as plt
import numpy as np

from cv_tools import get_cdf

spaces = ('hsv', 'hls', 'lab')
spaces_transform = {'hls': cv2.COLOR_BGR2HLS,
                    'hsv': cv2.COLOR_BGR2HSV,
                    'lab': cv2.COLOR_BGR2LAB}
spaces_invtransf = {'hls': cv2.COLOR_HLS2BGR,
                    'hsv': cv2.COLOR_HSV2BGR,
                    'lab': cv2.COLOR_LAB2BGR}
spaces_names = {'hls': ('Hue', 'Luminosity', 'Saturation'),
                'hsv': ('Hue', 'Saturation', 'Value'),
                'lab': ('L', 'A', 'B')}
spaces_correct_channel = {'hls': 1,
                          'hsv': 2,
                          'lab': 0}


def main():

    space = sys.argv[1]
    path_image = sys.argv[2]

    arr_image = cv2.imread(path_image, cv2.IMREAD_COLOR)

    space_transf = spaces_transform[space]
    img_hsv = cv2.cvtColor(arr_image, space_transf)
    h, s, v = cv2.split(img_hsv)

    channels = spaces_names[space]
    print('\n')

    colors = ('k', 'brown', 'b', 'g')

    fig = plt.figure()
    fig.canvas.set_window_title(os.path.basename(path_image))
    for i, (title, ch) in enumerate(zip(channels, (h, s, v))):
        plt.subplot(1, 3, i+1)
        plt.hist(ch.ravel(), 256, [0,256])
        for i, pct in enumerate(get_percentiles(ch, [5, 25, 75, 95])):
            plt.axvline(x = pct, color=colors[i], ls='dashed')
        plt.title(title)
    plt.show()
    plt.ion()


def get_percentiles(channel, pcts=[5, 95]):

    cdf = get_cdf(channel)
    return [ np.abs(cdf - pct*1e-2).argmin() for pct in pcts ]


if __name__ == "__main__":
    main()
