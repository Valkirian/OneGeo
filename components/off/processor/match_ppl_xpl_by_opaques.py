#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import argparse
import multiprocessing as mp
import os
import sys

import numpy as np
from skimage.filters import scharr

from common import (DebugLog, ensure_dir)
import cv_tools as cvt
from dft_stitch import (get_phasecorr_peak, warp_image,
                        get_translation_matrix, transform_points,
                        get_bounding_box)

debug_log = DebugLog()

image_kinds = {'xpl', 'ppl'}


def main():

    opt = process_command_line()
    print opt

    ensure_dir(opt.work_dir)

    opaques, crop_size = gather_images_data(opt.files, opt.threads)

    delta_pair = [opaques['xpl'], opaques['ppl']]
    delta_xp = compute_displacements_direct(delta_pair, 0)[1]
    debug_log("PPL->XPL mismatch is", delta_xp)

    success = align_images(opt.files, delta_xp, opt.work_dir)

    result = "done" if success else "failed"
    debug_log("Registration job", result)


def gather_images_data(file_list, max_threads):

    crop_sets = { kind: sorted(fi for fi in file_list
                               if kind in fi and "crop" in fi)
                 for kind in image_kinds }

    # Get reference crops and bounding boxes, chosen to be the middle element
    pool = mp.Pool(processes=2)
    jobs = []
    for kind, files_list in crop_sets.items():
        debug_log("Reading crop images of kind", kind)
        jobs.append(pool.apply_async(opaque_builder,
                                     (kind, files_list)))
    pool.close()
    pool.join()
    opaques = dict(job.get() for job in jobs)
    crop_size = max(opaques['xpl'].shape)

    return opaques, crop_size


def align_images(file_list, delta_xp, work_dir):

    body_sets = { kind: sorted(fi for fi in file_list
                               if kind in fi and "crop" not in fi)
                 for kind in image_kinds }

    # Get image target size
    xpl_image = cvt.file_to_cv2(body_sets['xpl'][0])
    target_size = xpl_image.shape[:2][::-1]

    # Then, align all image files (except the reference) and crop to common box
    pool = mp.Pool(processes=4)
    jobs = []
    for target in body_sets['ppl']:
        jobs.append(pool.apply_async(image_align,
                                    (target, delta_xp, target_size, work_dir)))
    pool.close()
    pool.join()
    success = all(job.get() for job in jobs)

    return success


def opaque_builder(kind, file_list, threshold=12):

    img_max = cvt.file_to_cv2(file_list[0]).max(axis=2)
    img_opaque = (img_max < threshold)

    for img_file in file_list:
        img_max = cvt.file_to_cv2(img_file).max(axis=2)
        img_opaque &= (img_max < threshold)

    return (kind, img_opaque)


def compute_displacements_direct(crops, reference_index):

    displacements = []
    crop_ref = crops[reference_index]

    for i, crop in enumerate(crops):

        if i == reference_index:
            displacements.append(np.r_[0, 0])
            continue

        debug_log("Computing translation of", i, "relative to", reference_index)
        peak_loc, peak_val = get_phasecorr_peak(crop_ref, crop, 100)

        displacements.append(peak_loc)
        debug_log("Translation (row, col) is", peak_loc, "value:", peak_val)

    return displacements


def image_align(img_file, delta, target_size, work_dir):

    basename = os.path.basename(img_file)
    cf_name = "reg-" + basename
    aligned_file = os.path.join(work_dir, cf_name)

    warp_matrix = get_translation_matrix(delta)[:2]
    img = cvt.file_to_cv2(img_file)
    img_aligned = warp_image(img, warp_matrix, target_size)
    success = cvt.cv2_to_file(img_aligned, aligned_file)

    if success:
        center_crop_box = get_center_crop_bbox(img_aligned, 1024)
        center_crop = crop_to_bounding_box(img_aligned, center_crop_box)
        center_crop_name = "crop-" + cf_name.replace(".png", ".jpg")
        center_crop_file = os.path.join(work_dir, center_crop_name)
        cvt.cv2_to_file(center_crop, center_crop_file)

    result = "done" if success else "failed"
    debug_log("Alignment of", img_file, "into", aligned_file, result)

    return success


def crop_to_bounding_box(image, bounding_box):

    min_x, max_x = sorted(co[0] for co in bounding_box)
    min_y, max_y = sorted(co[1] for co in bounding_box)

    im_roi = image[min_y:max_y, min_x:max_x]

    return im_roi


def get_center_crop_bbox(image, crop_size):

    center = get_bounding_box(image.shape)[1]/2
    center_crop_delta = np.r_[crop_size, crop_size]/2

    corner_start = center - center_crop_delta
    corner_end = center + center_crop_delta
    crop_box = [corner_start, corner_end]

    return crop_box


def process_command_line():

    description = ("Finds the optimal common registration among a set of images"
                   "by computing relative translations")
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", nargs='*',
            help="Input image files to be registered")
    parser.add_argument('-c', "--crop-size", type=int, default=2048,
            help="image crop side length for translation estimation")
    parser.add_argument('-d', "--work-dir", default='/dev/shm/',
            help="Where to write generated files to")
    parser.add_argument('-k', "--keep-uncropped", action="store_true",
            help="Avoids erasing of aligned uncropped output images")
    parser.add_argument('-t', "--threads", type=int,
                        default=max(1, mp.cpu_count() - 1),
            help=("Maximum number of simultaneous processes to execute"))

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
