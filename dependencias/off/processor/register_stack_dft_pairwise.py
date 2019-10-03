#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import argparse
import os
import sys

import cv2
import numpy as np

#import multiprocessing as mp
#import sharedmem as shm

from common import (DebugLog, ensure_dir)
import cv_tools as cvt
from dft_stitch import (get_phasecorr_peak, get_translation_matrix,
                        transform_points)

debug_log = DebugLog()

#TODO match a ppl and an xpl with SIFT.
#TODO fix cropping of images
def main():

    opt = process_command_line()
    print opt

    ensure_dir(opt.work_dir)

    targets = opt.files
    first_img = opt.files[0]
    basename_ref = os.path.basename(first_img)

    img1 = cvt.file_to_cv2(first_img)
    img1_crop = cvt.color_to_gray(crop_center_chunk(img1, opt.crop_size))
    img1_bb = get_bounding_box(img1.shape[:2])

    all_bboxes = [img1_bb,]
    relative_displacements = []

    # Get pairwise relative displacements
    for img_file in targets[1:]:

        basename = os.path.basename(img_file)
        img2 = cvt.file_to_cv2(img_file)
        img2_crop = cvt.color_to_gray(crop_center_chunk(img2, opt.crop_size))

        debug_log("Computing translation of", basename, "relative to",
                  basename_ref)

        peak_loc, peak_val = get_phasecorr_peak(img1_crop, img2_crop, 100)
        debug_log("Translation is", peak_loc, "value:", peak_val)

        relative_displacements.append(peak_loc)
        all_bboxes.append(get_bounding_box(img2.shape[:2]))

        img1, img1_crop, basename_ref = img2, img2_crop, basename

    del img1, img2, img1_crop, img2_crop

    # Determine largest bounding box
    bboxes_area = np.array([get_bounding_box_area(bbox) for bbox in all_bboxes])
    largest_area = np.argmax(bboxes_area)
    largest_bbox = all_bboxes[largest_area]
    target_size = get_bounding_box_size(largest_bbox)
    reference = targets[largest_area]
    basename_ref = os.path.basename(reference)

    print "disps:", relative_displacements
    debug_log("Largest area image is", reference, "({})".format(largest_area))

    # Propagate displacements
    pre_aligned_files = []
    for i, img_file in enumerate(targets):

        # Displacements are applied relative to largest bbox
        if i == largest_area:
            pre_aligned_files.append(reference)
            continue

        basename = os.path.basename(img_file)
        img = cvt.file_to_cv2(img_file)

        # displacement[i] = pos[i+1] - pos[i]
        if largest_area < i:
            disp_chain = range(largest_area, i)
            direction = 1
        else:
            disp_chain = range(i, largest_area)
            direction = -1

        total_displacement = direction * sum(relative_displacements[j]
                                             for j in disp_chain)
        debug_log("Displacement from", reference, "to", img_file, "is",
                  total_displacement)
        print "dir", direction, "; chain", disp_chain
        warp_matrix = get_translation_matrix(total_displacement)[:2]

        img_aligned = align_image(img, warp_matrix, target_size)
        aligned_file = os.path.join(opt.work_dir, "pre-" + basename)
        success = cvt.cv2_to_file(img_aligned, aligned_file)
        if success:
            pre_aligned_files.append(aligned_file)

        result = "done" if success else "failed"
        debug_log("Alignment of", img_file, "into", aligned_file, result)

    common_box = intersect_bounding_boxes(target_size, all_bboxes)

    for fi_aligned in pre_aligned_files:
        debug_log("Cropping", fi_aligned, newline=False)
        aligned = cvt.file_to_cv2(fi_aligned)
        cropped = crop_to_bounding_box(aligned, common_box)

        cf_name = (("reg-" + basename_ref) if fi_aligned == reference else
                   os.path.basename(fi_aligned).replace("pre-", "reg-"))
        cropped_file = os.path.join(opt.work_dir, cf_name)
        success = cvt.cv2_to_file(cropped, cropped_file)

        if success:
            center_crop = crop_center_chunk(cropped, 1024)
            center_crop_name = "crop-" + cf_name.replace(".png", ".jpg")
            center_crop_file = os.path.join(opt.work_dir, center_crop_name)
            cvt.cv2_to_file(center_crop, center_crop_file)

            if not opt.keep_uncropped and fi_aligned != reference:
                os.remove(fi_aligned)

        result = "done" if success else "failed"
        print(result)


def align_image(image, warp_matrix, target_size):

    aligned = cv2.warpAffine(image, warp_matrix, tuple(target_size),
                             flags=(cv2.INTER_LINEAR))
    return aligned


def get_adjusted_bounding_box(ref_bb, target_bb, warp_matrix):

    target_warped_bb = transform_points(target_bb, warp_matrix)
    adjusted_bb = adjust_bounding_box(ref_bb, target_warped_bb)

    return adjusted_bb


def get_bounding_box(shape):

    sizes = shape[:2]
    return np.array([[0, 0], [sizes[1], 0], [0, sizes[0]], list(sizes[::-1])])


def get_bounding_box_area(bounding_box):

    return np.prod(get_bounding_box_size(bounding_box))


def get_bounding_box_size(bounding_box):

    return (bounding_box[3] - bounding_box[0])


def adjust_bounding_box(ref_bb, target_bb):

    new_bb = np.zeros_like(ref_bb)
    ref_range = [np.sort(np.unique(ref_bb[:, 0])),
                 np.sort(np.unique(ref_bb[:, 1]))]

    tgt_range = [first_and_last(np.sort(np.unique(target_bb[:, 0]))),
                 first_and_last(np.sort(np.unique(target_bb[:, 1])))]

    for i, (ref_point, target_point) in enumerate(zip(ref_bb, target_bb)):
        for dim in (0, 1):
            tgt_inside_ref = ref_range[dim][0] <= target_point[dim] <= ref_range[dim][1]
            ref_inside_tgt = tgt_range[dim][0] <= ref_point[dim] <= tgt_range[dim][1]

            if tgt_inside_ref:
                new_bb[i, dim] = target_point[dim]
            elif ref_inside_tgt:
                new_bb[i, dim] = ref_point[dim]
            else:
                print("Weird case at dim {}, index {}".format(dim, i))

    return new_bb


def first_and_last(array):

    return [array[0], array[-1]]


def intersect_bounding_boxes(original_size, bounding_boxes):

    absolute_bb = []

    for index in range(4):
        absolute_bb.append([-1, -1])
        for dim in (0, 1):
            max_coord = max(bb[index][dim] for bb in bounding_boxes)
            fun = max if max_coord < original_size[dim]/2 else min
            absolute_bb[index][dim] = fun(bb[index][dim] for bb in bounding_boxes)

    return absolute_bb


def crop_to_bounding_box(image, bounding_box):

    _, min_x, max_x, _ = sorted(co[0] for co in bounding_box)
    _, min_y, max_y, _ = sorted(co[1] for co in bounding_box)

    im_roi = image[min_y:max_y, min_x:max_x]

    return im_roi


def crop_center_chunk(image, chunk_length=1024):

    center = np.array(image.shape[:2])/2
    center_crop_delta = np.r_[chunk_length, chunk_length]/2
    corner_start = center - center_crop_delta
    corner_end = center + center_crop_delta
    center_crop = image[corner_start[0]:corner_end[0],
                        corner_start[1]:corner_end[1]]
    return center_crop


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

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
