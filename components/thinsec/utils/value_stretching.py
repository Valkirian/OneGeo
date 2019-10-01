import os
import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt


def main():

    """
    clahe_tile_size = int(sys.argv[1])
    clahe_clip_limit = float(sys.argv[2])
    """
    image_file = sys.argv[1]
    upper_pct = float(sys.argv[2]) if len(sys.argv) > 2 else 99
    lower_pct = float(sys.argv[3]) if len(sys.argv) > 3 else 1

    img = cv2.imread(image_file, cv2.IMREAD_COLOR)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    """ Plot Histogram of value channel
    """
    plt.figure()
    for i, (title, ch) in enumerate(zip(('h', 's', 'v'), (h, s, v))):
        plt.subplot(1, 3, i+1)
        plt.hist(ch.ravel(), 256, [0,256])
        plt.title(title)
    plt.show()
    plt.ion()

    """
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit,
                            tileGridSize=(clahe_tile_size, clahe_tile_size))
    v_equ = clahe.apply(v)
    v_equ = cv2.equalizeHist(v)
    img_equ_hsv = cv2.merge([h, s, v_equ])
    img_equ = cv2.cvtColor(img_equ_hsv, cv2.COLOR_HSV2BGR)

    base_name = os.path.splitext(os.path.basename(image_file))[0]
    image_equ_file = os.path.join("/dev/shm/",
                                  base_name + ".png")
    cv2.imwrite(image_equ_file, img_equ)
    print(image_equ_file)
    """

    """
    img_out = os.path.abspath(os.path.splitext(os.path.basename(image_file))[0]
                              + "-stretch.jpg")
    cv2.imwrite(img_out, stretch_value(img, lower_pct, upper_pct))
    print("value-stretched image written to: {}".format(img_out))
    """


def stretch_value(image, lower_pct=1, upper_pct=99):

    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    v_hist = cv2.calcHist([v], [0], None, [256], [0, 256])
    v_cdf = v_hist.cumsum()
    v_cdf_norm = v_cdf/v_cdf.max()

    v_lower = np.abs(v_cdf_norm - lower_pct*1e-2).argmin()
    v_upper = np.abs(v_cdf_norm - upper_pct*1e-2).argmin()
    v_corr_stretch = 0 + ((v.astype(float) - v_lower)*(v.max()-v.min()))/(v_upper-v_lower)
    v_corr = v_corr_stretch.round().astype(int)
    v_corr[v_corr < 0] = 0
    v_corr[v_corr > 255] = 255
    img_equ_hsv = cv2.merge([h, s, v_corr.astype('uint8')])
    img_equ = cv2.cvtColor(img_equ_hsv, cv2.COLOR_HSV2BGR)

    return img_equ

if __name__ == "__main__":
    main()
