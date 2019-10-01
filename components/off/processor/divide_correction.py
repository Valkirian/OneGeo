#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import itertools
import multiprocessing as mp
import os

import cv2
import numpy as np

from common import (parse_image_grid_list, ensure_dir, img_data_fmt)
from cv_tools import (image_load_resize, get_cdf)


def main():

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("operation", choices=["compute", "apply"],
        help = "Operation to perform")
    ap.add_argument("files", nargs='*',
        help = "Input files")
    ap.add_argument("-b", "--blankfield", default=None,
        help = "Blank field file")
    ap.add_argument("-r", "--resize", default=100, type=int,
        help = "Resize percentage")
    ap.add_argument("-u", "--blur-alpha", default=0, type=float,
        help = "Blur strength from 0 to 1")
    ap.add_argument("-o", "--output", default=os.path.abspath(os.path.curdir),
        help = "output path")
    ap.add_argument('-t', "--threads", type=int,
                    default=max(1, mp.cpu_count() - 1),
            help=("Maximum number of simultaneous processes to execute"))
    """
    ap.add_argument("-d", "--dark", action="store_true",
        help = "Whether to consider images as having a dark blankfield")
    """

    args = ap.parse_args()
    print(args)

    if args.operation == "compute":
        bfield = generate_blankfield(args.files, int(args.resize), args.threads)
        ensure_dir(args.output)
        cv2.imwrite(os.path.join(args.output, "blankfield-stat.png"),
                    bfield)

    elif args.operation == "apply":
        apply_blankfield(args.files, args.blankfield, args.output,
                         int(args.resize), args.blur_alpha, args.threads)
                         #int(args.resize), args.dark)


def generate_write_blankfield(files_in, output_dir, percentage=100, threads=1):

    bfield = generate_blankfield(files_in, int(percentage), threads)
    ensure_dir(output_dir)
    success = cv2.imwrite(os.path.join(output_dir, "blankfield-stat.png"),
                          bfield)
    return success


def get_percentile(im, percentile=[50,]):

    if len(im.shape) > 2:
        im_hist = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        im_hist = im

    cdf = get_cdf(im_hist)
    v_percentile = [np.abs(cdf - pct*1e-2).argmin() for pct in percentile]
    if len(v_percentile) == 1:
        v_percentile = v_percentile[0]

    return v_percentile


def divide_blankfield(im, blank_field):

    im_div = im.astype(float) / blank_field
    rescale = im.max() / im_div.max()
    return np.uint8(rescale * im_div)


def get_blankfield_value(ims):

    accum = ims.next().astype(float)
    n_obj = 1

    for image in ims:
        accum += image.astype(float)
        n_obj += 1

    return (accum/n_obj).astype('uint8')


# TODO parallelize this
def generate_blankfield(files, percentage=100, threads=1):

    n_rows, n_cols, row_cells, _, _, _, _ = parse_image_grid_list(files)
    cell_num = ( (i, j, fi) for i, row in enumerate(row_cells)
                  for j, fi in enumerate(row) )
    bulk_ims = ( image_load_resize(fi, percentage) for i, j, fi in cell_num
                if 0 < i < n_rows-1 and 0 < j < n_cols-1 )
    blank_field = get_blankfield_value(bulk_ims)

    return blank_field


def apply_blankfield(files, blank_field_file, dest_dir, percentage=100,
                     blur_alpha=0.7, threads=1, is_dark=False):

    files = sorted(files)

    ensure_dir(dest_dir)

    blank_field = image_load_resize(blank_field_file, percentage)
    if blur_alpha > 0:
        blank_field = omomorphic_shading_extraction(blank_field, blur_alpha)

    blankfield_wgt = get_blankfield_weights(blank_field)

    ims = (image_load_resize(i, percentage) for i in files)
    job_args = [ (blankfield_wgt, im_i, f, dest_dir) for f, im_i
                in itertools.izip(files, ims) ]
    if threads == 1:
        results = [ apply_blankfield_weights(*args) for args in job_args ]
    else:
        pool = mp.Pool(processes=threads)
        jobs = [ pool.apply_async(apply_blankfield_weights, args) for args
                in job_args ]
        pool.close()
        pool.join()
        results = [ job.get() for job in jobs ]
    success = all(results)

    return success


def omomorphic_shading_extraction(image, alpha=0.7):

    h, w = image.shape[:2]
    l = 2*int(0.5 * alpha * min(h, w)) + 1

    ln_image = np.log1p(image, dtype=np.float32)
    blurred = cv2.GaussianBlur(ln_image, (l, l), 0)
    bfield = np.expm1(blurred)

    return bfield.round().astype(np.uint8)


def get_blankfield_weights(blankfield):

    blankfield_mean = blankfield.mean(axis=0).mean(axis=0)
    blankfield_wgt = blankfield_mean[None, None, :]/blankfield

    return blankfield_wgt


def apply_blankfield_weights(blankfield_wgt, image, filename, destination_dir):

    """
    if is_dark and is_glass(im_i, 80):
        im_ii = blankfield_weighted_correct(im_i, blank_field_weight)
    else:
        im_ii = divide_blankfield(im_i, blank_field)
    """
    image_safe = 0.5 * image
    im_corr = blankfield_wgt * image_safe
    rescale = image.max() / im_corr.max()
    im_ii = ( rescale * im_corr ).round().astype(np.uint8)

    save_file = os.path.splitext(os.path.basename(filename))[0] + '.' + img_data_fmt
    success = cv2.imwrite(os.path.join(destination_dir, save_file), im_ii)

    return success


def is_glass(im, diff=25):

    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h, _, _ = cv2.split(im_hsv)
    v_percentile = get_percentile(h, percentile=[5, 95])

    return ((v_percentile[1] - v_percentile[0]) <= diff)


if __name__ == "__main__":
    main()
