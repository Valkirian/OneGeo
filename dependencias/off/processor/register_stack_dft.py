#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import argparse
import os
import sys

import cv2
import numpy as np

from common import (DebugLog, ensure_dir)
import cv_tools as cvt
from dft_stitch import (get_phasecorr_peak, get_translation_matrix,
                        transform_points, get_bounding_box)

debug_log = DebugLog()

def main():

    opt = process_command_line()
    print opt

    ensure_dir(opt.work_dir)

    band = opt.crop_size

    basename_ref = os.path.basename(opt.reference)
    img1 = cvt.file_to_cv2(opt.reference)
    img1_bb = get_bounding_box(img1.shape)
    img1_br = img1_bb[1]
    img1_crop_bb = [(img1_br - band)/2, (img1_br + band)/2]
    target_size = get_bounding_box_size(img1_bb)

    img1_crop = cvt.color_to_gray(crop_to_bounding_box(img1, img1_crop_bb))

    """
    Using crop chunks centered at each images' center impedes resolving
    the image-to-image displacement
    """
    #img1_crop = cvt.color_to_gray(crop_center_chunk(img1, band))

    all_bboxes = [img1_bb.tolist(),]
    pre_aligned_files = [opt.reference,]

    # trainImages
    targets = sorted(set(opt.files) - set([opt.reference,]))
    for img_file in targets:

        basename = os.path.basename(img_file)
        img2 = cvt.file_to_cv2(img_file)
        img2_crop = cvt.color_to_gray(crop_to_bounding_box(img2, img1_crop_bb))
        #img2_crop = cvt.color_to_gray(crop_center_chunk(img2, band))

        debug_log("Computing translation of", basename, "relative to",
                  basename_ref)

        peak_loc, peak_val = get_phasecorr_peak(img1_crop, img2_crop, 100)
        debug_log("Translation is", peak_loc, "value:", peak_val)
        warp_matrix = get_translation_matrix(peak_loc)[:2]

        img2_aligned = warp_image(img2, warp_matrix, target_size)
        img2_adj_bb = get_adjusted_bounding_box(img1_bb,
                                                get_bounding_box(img2.shape),
                                                warp_matrix)
        aligned_file = os.path.join(opt.work_dir, "pre-" + basename)
        cvt.cv2_to_file(img2_aligned, aligned_file)
        all_bboxes.append(img2_adj_bb)
        pre_aligned_files.append(aligned_file)

        debug_log("Alignment of", img_file, "done")

    common_box = intersect_bounding_boxes(all_bboxes)

    for fi_aligned in pre_aligned_files:
        debug_log("Cropping", fi_aligned, newline=False)
        aligned = cvt.file_to_cv2(fi_aligned)
        cropped = crop_to_bounding_box(aligned, common_box)

        cf_name = (("reg-" + basename_ref) if fi_aligned == opt.reference else
                   os.path.basename(fi_aligned).replace("pre-", "reg-"))
        cropped_file = os.path.join(opt.work_dir, cf_name)
        success = cvt.cv2_to_file(cropped, cropped_file)

        if success:
            center_crop = crop_center_chunk(cropped, 1024)
            center_crop_name = "crop-" + cf_name.replace(".png", ".jpg")
            center_crop_file = os.path.join(opt.work_dir, center_crop_name)
            cvt.cv2_to_file(center_crop, center_crop_file)

            if not opt.keep_uncropped and fi_aligned != opt.reference:
                os.remove(fi_aligned)

        result = "done" if success else "failed"
        print(result)


def get_adjusted_bounding_box(ref_bb, target_bb, warp_matrix):

    target_warped_bb = transform_points(target_bb, warp_matrix)
    adjusted_bb = adjust_bounding_box(ref_bb, target_warped_bb)

    return adjusted_bb


def warp_image(image, warp_matrix, target_size):

    size_x, size_y = target_size
    aligned = cv2.warpAffine(image, warp_matrix, (size_x, size_y),
                             flags=(cv2.INTER_LINEAR))
    return aligned


def get_bounding_box_size(bounding_box):

    return (bounding_box[1] - bounding_box[0])


def get_bounding_box_area(bounding_box):

    return np.prod(get_bounding_box_size(bounding_box))


def adjust_bounding_box(ref_bb, target_bb):

    new_bb = target_bb.copy()

    for dim in (0, 1):
        # top-left corner
        if target_bb[0][dim] < ref_bb[0][dim]:
            new_bb[0][dim] = ref_bb[0][dim]

        # bottom-right corner
        if target_bb[1][dim] > ref_bb[1][dim]:
            new_bb[1][dim] = ref_bb[1][dim]

    return new_bb


def intersect_bounding_boxes(bounding_boxes):

    tl_x, tl_y = [max(bb[0][dim] for bb in bounding_boxes) for dim in (0,1)]
    br_x, br_y = [min(bb[1][dim] for bb in bounding_boxes) for dim in (0,1)]

    return np.array([[tl_x, tl_y], [br_x, br_y]])


def join_bounding_boxes(bounding_boxes):

    tl_x, tl_y = [min(bb[0][dim] for bb in bounding_boxes) for dim in (0,1)]
    br_x, br_y = [max(bb[1][dim] for bb in bounding_boxes) for dim in (0,1)]

    return np.array([[tl_x, tl_y], [br_x, br_y]])


def crop_to_bounding_box(image, bounding_box):

    min_x, max_x = sorted(co[0] for co in bounding_box)
    min_y, max_y = sorted(co[1] for co in bounding_box)

    im_roi = image[min_y:max_y, min_x:max_x]

    return im_roi


def crop_center_chunk(image, chunk_length=1024):

    center = get_bounding_box(image.shape)[1]/2
    center_crop_delta = np.r_[chunk_length, chunk_length]/2

    corner_start = center - center_crop_delta
    corner_end = center + center_crop_delta

    return crop_to_bounding_box(image, [corner_start, corner_end])


def process_command_line():

    description = ("Finds the optimal common registration among a set of images"
                   "by computing relative translations")
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("reference", type=str,
            help="Reference image against which the input images will be aligned")
    parser.add_argument("files", nargs='*',
            help="Input image files to be registered")
    parser.add_argument('-c', "--crop-size", type=int, default=2048,
            help="image crop side length for translation estimation")
    parser.add_argument('-d', "--work-dir", default='/dev/shm/',
            help="Where to write generated files to")
    parser.add_argument('-k', "--keep-uncropped", action="store_true",
            help="Avoids erasing of aligned uncropped output images")

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
