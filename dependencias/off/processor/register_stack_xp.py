#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import argparse
import itertools as it
import multiprocessing as mp
import sys

import cv2
import numpy as np

from common import (DebugLog, ensure_dir)
import cv_tools as cvt
from dft_stitch import (get_phasecorr_peak, get_phasecorr_peak__old,
                        get_translation_matrix, transform_points)
import register_stack_dft_mp as reg

debug_log = DebugLog()


def main():

    opt = process_command_line()
    print opt

    ensure_dir(opt.work_dir)

    success = register_images(opt.files, opt.crop_size, opt.threads,
                              opt.work_dir, opt.use_borders,
                              opt.first_image_is_absolute,
                              make_jpeg=opt.write_also_jpeg)

    result = "done" if success else "failed"
    debug_log("Registration job", result)


def register_images(file_list, crop_size, max_threads, work_dir,
                    use_borders=False, first_image_is_absolute=True,
                    make_center_chunk=True, target_files_map={}, make_jpeg=False,
                    assume_zero_displacements=True):

    data = gather_images_data(file_list, crop_size, max_threads,
                              assume_zero_displacements=assume_zero_displacements)

    if target_files_map:
        data['tgtpaths'] = target_files_map

    delta_xp = match_ppl_xpl(data, crop_size, False, use_borders)
    debug_log("PPL->XPL mismatch is", delta_xp)
    success = align_images(data, delta_xp, work_dir, first_image_is_absolute,
                           max_threads, make_center_chunk, make_jpeg)

    return success


def gather_images_data(file_list, crop_size, max_threads,
                       assume_zero_displacements=True):

    debug_log("Gathering images' regions")
    image_classes = {'xpl', 'ppl'}
    files_list = { cl: sorted(fi for fi in file_list if cl in fi)
                  for cl in image_classes }

    xpl_data = reg.gather_images_data(files_list['xpl'], crop_size, max_threads,
                                      False, None, assume_zero_displacements)
    xpl_reference_corner = xpl_data['bboxes'][0][1]

    ppl_data = reg.gather_images_data(files_list['ppl'], crop_size, max_threads,
                                      False, xpl_reference_corner, assume_zero_displacements)

    return {'xpl': xpl_data, 'ppl': ppl_data}


def match_ppl_xpl(data, crop_size, intersect=True, use_borders=True):

    debug_log("Registering images across classes")
    ppl_ref, ppl_ref_bbox, ppl_ref_shift = get_aggregated_crops(data['ppl'],
                                                                crop_size,
                                                                intersect,
                                                                use_borders,
                                                                (100, 120))
    xpl_ref, xpl_ref_bbox, xpl_ref_shift = get_aggregated_crops(data['xpl'],
                                                                crop_size,
                                                                intersect,
                                                                use_borders,
                                                                (100, 300))

    tgt_sz = np.fmin(ppl_ref.shape, xpl_ref.shape)[::-1]
    common_box = intersect_bounding_boxes(tgt_sz, [ppl_ref_bbox, xpl_ref_bbox])
    common_shift = np.array(common_box[0][::-1])

    xpl_ref_common = crop_to_bounding_box(xpl_ref, common_box)
    ppl_ref_common = crop_to_bounding_box(ppl_ref, common_box)

    delta_pair = [xpl_ref_common, ppl_ref_common]
    if not use_borders:
        delta_pair = map(cvt.color_to_gray, delta_pair)

    print xpl_ref_shift, ppl_ref_shift, common_shift

    delta_xp = (xpl_ref_shift - ppl_ref_shift + common_shift +
                compute_displacements_direct(delta_pair, 0)[1])

    return delta_xp


def align_images(data, delta_xp, work_dir, first_image_is_absolute=True,
                 max_threads=4, make_center_chunk=True, make_jpeg=False):

    # Update ppl class displacements to match xpl class
    data['ppl']['deltas'] = [ delta + delta_xp for delta in data['ppl']['deltas'] ]

    # Flatten data dictionary
    keys = data['xpl'].keys()
    flattened_data = { key: data['xpl'][key] + data['ppl'][key] for key in keys }

    if 'tgtpaths' in data:
        flattened_data['tgtpaths'] = data['tgtpaths']

    return reg.align_images(flattened_data, work_dir, first_image_is_absolute,
                            max_threads, make_center_chunk, make_jpeg)


def get_aggregated_crops(data_dict, crop_size, intersect=True, use_borders=True,
                         borders_param=(100, 300), ref_index=0):

    target_size = data_dict['crops'][ref_index].shape[:2][::-1]
    chunk_bbox = get_bounding_box(2*[crop_size])

    shifts_crops = it.izip(data_dict['deltas'], data_dict['crops'])
    mats_crops = ( (get_translation_matrix(shift)[:2], chnk) for shift, chnk
                  in shifts_crops )
    tld_crops = ( (warp_image(chnk, mat, target_size),
                   get_adjusted_bounding_box(chunk_bbox, chunk_bbox, mat))
                 for mat, chnk in mats_crops )
    to_add = ( list(tld_crops) if not use_borders else
               [ (reg.get_borders(chnk, *borders_param), bbox) for chnk, bbox
                in tld_crops ] )
    stack = np.stack( crop for crop, _ in to_add )
    common_box = intersect_bounding_boxes(target_size,
                                          [bbox for _, bbox in to_add])

    agg = (stack/255).prod(axis=0) if intersect else stack.mean(axis=0)
    dyn_range = agg.max() - agg.min()
    agg_stretched = (( 255*(agg - agg.min()) / dyn_range) if dyn_range > 1 else
                      agg )

    agg_common = crop_to_bounding_box(agg_stretched, common_box)

    aggregated = agg_common.astype(np.uint8)
    agg_bbox = get_bounding_box(agg_common.shape)
    agg_shift = np.array(common_box[0][::-1])

    return aggregated, agg_bbox, agg_shift


def compute_displacements_direct(crops, reference_index, use_old=False):

    displacements = []
    crop_ref = crops[reference_index]

    for i, crop in enumerate(crops):

        if i == reference_index:
            displacements.append(np.r_[0, 0])
            continue

        print "Computing translation (row, col) of", i, "relative to", reference_index, 

        if use_old:
            peak_loc, peak_val = get_phasecorr_peak__old(crop_ref, crop)
        else:
            peak_loc, peak_val = get_phasecorr_peak(crop_ref, crop, 100)

        displacements.append(peak_loc)
        print ":", peak_loc, "value:", peak_val

    return displacements


def warp_image(image, warp_matrix, target_size):

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


def process_command_line():

    description = ("Finds the optimal common registration among a set of images"
                   "by computing relative translations")
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", nargs='+',
            help="Input image files to be registered")
    parser.add_argument('-c', "--crop-size", type=int, default=2048,
            help="image crop side length for translation estimation")
    parser.add_argument('-d', "--work-dir", default='/dev/shm/',
            help="Where to write generated files to")
    parser.add_argument('-b', "--use-borders", action='store_true',
            help="Perform registration using the edge-detection of images")
    parser.add_argument('-f', "--first-image-is-absolute", action='store_true',
            help="Align images with respect to reference, and crop them "
                 "to its size")
    parser.add_argument('-t', "--threads", type=int,
                        default=max(1, mp.cpu_count() - 1),
            help=("Maximum number of simultaneous processes to execute"))
    parser.add_argument('-j', "--write-also-jpeg", action='store_true',
            help="Create JPEG versions of the registered images")

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
