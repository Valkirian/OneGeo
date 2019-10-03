#!/usr/bin/env python2

import multiprocessing
import os

import cv2
import numpy as np

from cv_tools import (get_cdf, image_resize, file_to_cv2,
                      stretch_value_channel, simple_blur, cv2_to_file)

def create_mosaic(files, im_out='mosaic', d_space=10, resize_in=0.4,
                  resize_out=0.5, stretch=True, **kwargs):
    """
    im_out: image output name prefix
    d_space: space between images in pixels
    """
    files = sorted(files)

    (n_rows, n_cols, row_cells, img_type, rows,
        digits_row, digits_col) = parse_image_grid_list(files)

    pool = multiprocessing.Pool()
    """ Local value stretching is not globally optimal
    if stretch:
        img_jobs = [pool.apply_async(resize_stretch, (f, 100*resize_in, 5, 99))
                    for f in files]
    else:
        img_jobs = [pool.apply_async(image_load_resize, (f, 100*resize_in))
                    for f in files]
    """

    if 'bfield' in kwargs:
        bfield_file = kwargs['bfield']
        bfield_blurred = os.path.splitext(bfield_file)[0] + '-blurred.png'
        #blur_blankfield(bfield_file, bfield_blurred)
        img_jobs = [pool.apply_async(blankfield_remove, (f, bfield_blurred,
                                                         100*resize_in))
                    for f in files]
    else:
        img_jobs = [pool.apply_async(image_load_resize, (f, 100*resize_in))
                    for f in files]
    pool.close()
    pool.join()

    ims = [job.get() for job in img_jobs]
    h, w, z = ims[0].shape

    canvas = np.zeros((n_rows*(d_space + h),
                n_cols*(d_space + w), z), np.uint8)

    for i in xrange(n_rows):
        for j in xrange(n_cols):
            pos_i, pos_j = i*(h + d_space), j*(w + d_space)
            linear_id = j + i*n_cols
            canvas[pos_i:pos_i+h, pos_j:pos_j+w, :] = ims[linear_id]

    resized = ''
    if resize_out:
        canvas = cv2.resize(canvas, (0, 0), fx=resize_out, fy=resize_out)
        resized = '-{}pc'.format(100*resize_out)

    stretched = ''
    if stretch:
        params = (5, 99)
        canvas = stretch_value_channel(canvas, *params)
        stretched = '-{}l_{}u'.format(*params)

    bfielded = ''
    if 'bfield' in kwargs:
        bfielded = '-bfield'

    output = kwargs['output']
    if output is None:
        name_spec = "{}-{:02d}_{:02d}{}{}{}.jpg"
        output = os.path.abspath(name_spec.format(im_out, n_rows, n_cols,
                                                  resized, stretched, bfielded))
    cv2.imwrite(output, canvas)

    return n_rows, n_cols, row_cells, img_type, rows, digits_row, digits_col, output


def parse_image_grid_list(files):

    file_table = [(f, os.path.splitext(os.path.basename(f))[0].split('_'))
                  for f in files]
    as_ints = [map(int, f[1]) for f in file_table]
    n_cols = 1 + max(c[1] for c in as_ints) - min(c[1] for c in as_ints)
    n_rows = 1 + max(c[0] for c in as_ints) - min(c[0] for c in as_ints)
    assert n_rows * n_cols == len(files), "Missing input images for a full stitch"

    extensions = set(os.path.splitext(os.path.basename(f))[1] for f in files)
    assert len(extensions) == 1, "There is more than one image file type: {}".format(' '.join(extensions))
    extension = extensions.pop()[1:]

    all_digits_row, all_digits_col = zip(*[map(len, f[1]) for f in file_table])
    assert max(all_digits_row) == min(all_digits_row), "File naming is not consistent over rows"
    assert max(all_digits_col) == min(all_digits_col), "File naming is not consistent over columns"
    digits_row = min(all_digits_row)
    digits_col = min(all_digits_col)

    rows = sorted(list(set(f[1][0] for f in file_table)))
    row_cells = [[f[0] for f in file_table if f[1][0] == row]
                 for row in rows]

    return n_rows, n_cols, row_cells, extension, rows, digits_row, digits_col


def image_load_resize(image_file, percentage=100):

    image = cv2.imread(image_file)
    return image_resize(image)


""" Local value stretching is not globally optimal
def resize_stretch(img_file, resize, lower_pct, upper_pct):

    return stretch_value_channel(image_load_resize(img_file, resize), lower_pct, upper_pct)
"""


def blankfield_remove(image_file, bfield_file, reduction_pct=100):

    image = image_load_resize(image_file, reduction_pct)
    #bfield = image_load_resize(bfield_file, reduction_pct)
    b, g, r = cv2.split(image)

    if is_glass(image, 80):
        g_high = g.astype(int) + 50
        g_high[g_high > 255] = 255
        return cv2.merge((b, g_high.astype('uint8'), r))
    else:
        b_high = b.astype(int) + 50
        b_high[b_high > 255] = 255
        return cv2.merge((b_high.astype('uint8'), g, r))

    #return blankfield_dark_correct(image, bfield, pre_blurred=True)


def blur_blankfield(bfield_file, bfield_blurred_file):

    bfield = file_to_cv2(bfield_file)
    cv2_to_file(simple_blur(bfield), bfield_blurred_file, 0.01)


"""
def is_glass (im, pct_span=30):

    l, _, _ = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2LAB))
    cdf = get_cdf(l)
    pct_lo = np.abs(cdf - 5e-2).argmin()
    pct_hi = np.abs(cdf - 95e-2).argmin()
    span = (pct_hi - pct_lo)/2.5

    return (span < pct_span)

"""
def is_glass (im, diff=25):

    h, _, _ = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
    cdf = get_cdf(h)
    pct_lo = np.abs(cdf - 5e-2).argmin()
    pct_hi = np.abs(cdf - 95e-2).argmin()

    return (pct_hi - pct_lo) < diff


if __name__ == "__main__":

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", nargs='+', required = True,
                    help = "input files")
    ap.add_argument("-r", "--resize", default = None, required = False,
    	help = "resize percentage (e.g., 40%%)")
    ap.add_argument("-s", "--stretch", action="store_true",
                    help="Whether to perform value stretching on input images")
    ap.add_argument("-k", "--bfield", default = None, required = False,
                	help="path to dark blank field")
    ap.add_argument("-o", "--output", default = None, required = False,
                    help = "path of target output file")
    args = vars(ap.parse_args())

    if args["resize"]:
        args["resize"] = float(args["resize"].strip('%'))/100

    kwargs = {}
    if args["bfield"]:
        kwargs['bfield'] = args['bfield']

    kwargs['output'] = (None if args['output'] is None else
                        os.path.abspath(args['output']))

    (n_rows, n_cols, row_cells, img_type, rows, digits_row, digits_col,
        output) = create_mosaic(args["input"], resize_out=args["resize"],
                                stretch=args["stretch"], **kwargs)

    print("These files span {} rows by {} columns".format(n_rows, n_cols))
    print("The rows are named with {} digits; the columns with {}".format(digits_row, digits_col))
    print("The image type of these files is: {}".format(img_type))
    print("Mosaic saved successfully as: {}".format(output))
