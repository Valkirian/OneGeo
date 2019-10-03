
import cv2
import os

import numpy as np

from cv_tools import blankfield_dark_correct

def image_load_resize(image_file, percentage=100):

    image = cv2.imread(image_file)
    if percentage != 100:
        size = np.array(image.shape[:2])
        new_size = (percentage*size/100.0).astype(int)

        resized = cv2.resize(image, tuple(new_size[::-1]))
    else:
        resized = image

    return resized


def get_percentile(im, percentile=[50]):

    if len(im.shape) > 2:
        im_hist = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        im_hist = im

    v_hist = cv2.calcHist([im_hist], [0], None, [256], [0,256])
    v_cdf = v_hist.cumsum()
    v_cdf_norm = v_cdf/v_cdf.max()
    v_percentile = []
    for pct in percentile:
         v_percentile.append(np.abs(v_cdf_norm - pct*1e-2).argmin())

    if not len(v_percentile) > 1:
        v_percentile = v_percentile[0]

    return v_percentile


def sequence_percentile(files):

    res = []
    for f in sorted(files):
        im = cv2.imread(f)
        v_percentile = get_percentile(im, [5, 50, 95])

        res.append((os.path.basename(f), v_percentile))

    return res


def get_blankfield_value(ims):

    # ims = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in ims]

    bf = 1.0*np.sum(ims, axis=0)/len(ims)
    h, w, z = bf.shape
    h = int(0.3*h)
    bf = cv2.GaussianBlur(bf, (h, h), 0)
    # bf = cv2.merge((bf, bf, bf))

    return bf.astype(np.uint8)


def divide_blankfield(im, blank_field):
    im = cv2.divide(im.astype(np.float), blank_field.astype(np.float))
    return np.uint8((255/im.max())*im)


def rescaling_percentiles(pct_data, files, dest_dir, fact=0.7):

    p_ave = pct_data.mean(axis=0)
    rescale = pct_data[:, 3] < fact*p_ave[3]

    for f, p_i in zip(files[rescale], pct_data[rescale, :]):
        save_file = os.path.basename(f).replace('nobfc-', '')
        f = os.path.join(dest_dir, save_file)

        print(f, p_i)

        im_i = cv2.imread(f)

        ratio = (get_percentile(im_i) - p_i[2])/p_i[2]

        if ratio > 1.8:
            alpha = 0.6*(p_ave[4] - p_ave[0])/im_i.max()
        elif ratio > 1.6 and ratio <= 1.8:
            alpha = 0.9*(p_ave[4] - p_ave[0])/im_i.max()
        elif ratio > 1.2 and ratio <= 1.6:
            alpha = 1.2*(p_ave[4] - p_ave[0])/im_i.max()
        else:
            alpha = 1.4*(p_ave[4] - p_ave[0])/im_i.max()

        beta = 0
        # beta = 0.1*p_ave[0]

        im_i = np.uint8(alpha*im_i + beta)
        cv2.imwrite(f, im_i)


def dark_linear_correction(pct_data, files, ims, blank_field, dest_dir, fact=0.65):

    p_ave = pct_data.mean(axis=0)
    linear_fix = pct_data[:, 3] < fact*p_ave[3]

    for im_i, f in zip(ims[linear_fix], files[linear_fix]):
        print(im_i, f)
        im_ii = np.uint8(blankfield_dark_correct(im_i, blank_field) + 0.5*p_ave[1])
        save_file = os.path.basename(f).replace('nobfc-', '')
        f = os.path.join(dest_dir, save_file)
        cv2.imwrite(f, im_ii)


def is_glass(im, diff=25):
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h, _, _ = cv2.split(im_hsv)
    v_percentile = get_percentile(h, percentile=[5, 95])

    return v_percentile[1] - v_percentile[0] <= diff


def blankfield_correction(files, blank_field_file, dest_dir, percentage, is_dark):

    files = sorted(files)

    if not dest_dir:
        dest_dir = os.path.join(os.path.dirname(files[0]), 'bfc')

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    ims = [image_load_resize(i, percentage=percentage) for i in files]

    if not blank_field_file:
        blank_field = get_blankfield_value(ims)
        cv2.imwrite(os.path.join(dest_dir, 'blank_field.jpg'), blank_field)
    else:
        blank_field = cv2.imread(blank_field_file)
        if not np.array_equal(blank_field.shape, ims[0].shape):
            blank_field = image_load_resize(blank_field_file, percentage=percentage)

    for f, im_i in zip(files, ims):

        if is_glass(im_i) and is_dark:
            im_ii = np.zeros_like(im_i, dtype=np.uint8)
        else:
            im_ii = divide_blankfield(im_i, blank_field)

        save_file = os.path.basename(f).replace('nobfc-', '')
        cv2.imwrite(os.path.join(dest_dir, save_file), im_ii)

'''
    pct_data = []
    percentiles = [5, 25, 50, 75, 99.5]

    for f, im_i in zip(files, ims):
        pct_data.append(get_percentile(im_i, percentiles))
        im_ii = divide_blankfield(im_i, blank_field)

        save_file = os.path.basename(f).replace('nobfc-', '')
        cv2.imwrite(os.path.join(dest_dir, save_file), im_ii)

    # rescaling_percentiles(np.asarray(pct_data), np.asarray(files), dest_dir)
    dark_linear_correction(np.asarray(pct_data), np.asarray(files),
                            np.asarray(ims), blank_field, dest_dir)
'''


if __name__ == "__main__":

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", nargs='+', required=True,
        help = "input files")
    ap.add_argument("-b", "--blankfield", default=None,
        help = "Blank field file")
    ap.add_argument("-r", "--resize", default=100,
        help = "Resize percentage")
    ap.add_argument("-o", "--output", default=None,
        help = "output path")
    ap.add_argument("-d", "--dark", action="store_true",
        help = "Whether to consider images as having a dark blankfield")

    args = vars(ap.parse_args())

    blankfield_correction(args['input'], args['blankfield'], args['output'],
                            int(args['resize']), args['dark'])
