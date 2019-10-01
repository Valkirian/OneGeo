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


def main():

    opt = process_command_line()
    print opt

    ensure_dir(opt.work_dir)

    data = gather_images_data(opt.files, opt.crop_size, opt.threads)
    delta_xp = match_ppl_xpl_opaques(data, opt.crop_size, opt.work_dir)
    debug_log("PPL->XPL mismatch is", delta_xp)
    success = align_images(data, delta_xp, opt.work_dir)

    result = "done" if success else "failed"
    debug_log("Registration job", result)


def gather_images_data(file_list, crop_size, max_threads):

    image_kinds = {'xpl', 'ppl'}
    file_sets = { kind: sorted(fi for fi in file_list if kind in fi)
                 for kind in image_kinds }
    ref_index = {}

    # Get reference crops and bounding boxes, chosen to be the middle element
    pool = mp.Pool(processes=2)
    jobs = []
    for kind, files_list in file_sets.items():
        debug_log("Reading reference images of kind", kind)
        middle_index = int(max(0, (len(files_list) - 1)/2))
        ref_index[kind] = middle_index
        jobs.append(pool.apply_async(reference_grabber,
                                     (kind, files_list[middle_index], crop_size)))
    pool.close()
    pool.join()
    ref_data = dict(job.get() for job in jobs)

    # Read the file lists with a few threads as each image is potentially huge
    pool = mp.Pool(processes=4)
    jobs = []
    for kind, files_list in file_sets.items():

        crop_box = ref_data[kind]['cropbox']
        debug_log("Reading target images of kind", kind,
                  "with crop chunk at", crop_box)
        for i, img_file in enumerate(files_list):
            if i != ref_index[kind]:
                jobs.append(pool.apply_async(image_list_grabber,
                                             (kind, img_file, i, crop_box)))
    pool.close()
    pool.join()
    parts = [job.get() for job in jobs]

    # Create base data structure
    data = {}
    for kind, files_list in file_sets.items():

        data_list = [ (i, crop, bbox) for (in_kind, i, crop, bbox) in parts
                     if kind == in_kind ]
        data_list.append((ref_index[kind], ref_data[kind]['crop-ref'],
                          ref_data[kind]['bbox-ref']))
        sorted_data = sorted(data_list)

        data[kind] = {'crops': [ elem[1] for elem in sorted_data ],
                      'bboxes': [ elem[2] for elem in sorted_data ],
                      'targets': files_list,
                      'ref-index': ref_index[kind],
                      'count': len(files_list)}

    # Compute displacements w.r.t reference via the image crops just obtained
    pool = mp.Pool(processes=max_threads)
    jobs = [pool.apply_async(pairwise_displacement_worker, (i, ddict, kind))
            for kind, ddict in data.items() for i in range(1, ddict['count'])]
    pool.close()
    pool.join()
    pdisps = [job.get() for job in jobs]

    for kind in data.keys():

        reference_index = data[kind]['ref-index']
        kind_deltas = [(i, disp) for in_kind, i, disp in pdisps
                       if kind == in_kind]
        deltas = [delta for i, delta in sorted(kind_deltas)]
        deltas_ref = add_pairwise_displacements(deltas, reference_index)
        data[kind]['deltas'] = deltas_ref

        reference_target = os.path.basename(file_sets[kind][reference_index])
        debug_log("Displacements relative to", reference_target, deltas_ref)

    return data


def match_ppl_xpl(data, crop_size, work_dir):

    ppl_reference = scharr(data['ppl']['crops'][data['ppl']['ref-index']])
    xpl_ref, xpl_ref_bbox, xpl_ref_shift = get_registered_averaged(data['xpl'],
                                                                   crop_size)
    delta_pair = [xpl_ref,
                  crop_to_bounding_box(ppl_reference, xpl_ref_bbox)]
    delta_xp = xpl_ref_shift + compute_displacements_direct(delta_pair, 0)[1]

    ppl_reg_file = os.path.join(work_dir, "ref_dir_ppl.png")
    cvt.cv2_to_file(cvt.simple_grayscale_stretch(ppl_reference), ppl_reg_file)

    xpl_reg_file = os.path.join(work_dir, "ref_avg_xpl.png")
    cvt.cv2_to_file(xpl_ref, xpl_reg_file)

    return delta_xp


def match_ppl_xpl_new(data, crop_size, work_dir):

    xpl_ref, _, _ = get_registered_averaged(data['xpl'], crop_size)
    ppl_ref, _, _ = get_registered_averaged(data['ppl'], crop_size)
    delta_pair = [xpl_ref, ppl_ref]
    delta_xp = compute_displacements_direct(delta_pair, 0)[1]

    ppl_reg_file = os.path.join(work_dir, "ref_avg_ppl.png")
    cvt.cv2_to_file(ppl_ref, ppl_reg_file)

    xpl_reg_file = os.path.join(work_dir, "ref_avg_xpl.png")
    cvt.cv2_to_file(xpl_ref, xpl_reg_file)

    return delta_xp


def match_ppl_xpl_opaques(data, crop_size, work_dir):

    xpl_ref, _, _ = get_registered_opaques(data['xpl'], crop_size)
    ppl_ref, _, _ = get_registered_opaques(data['ppl'], crop_size)
    delta_pair = [xpl_ref, ppl_ref]
    delta_xp = compute_displacements_direct(delta_pair, 0)[1]

    ppl_reg_file = os.path.join(work_dir, "ref_opaque_ppl.png")
    cvt.cv2_to_file(ppl_ref, ppl_reg_file)

    xpl_reg_file = os.path.join(work_dir, "ref_opaque_xpl.png")
    cvt.cv2_to_file(xpl_ref, xpl_reg_file)

    return delta_xp


def align_images(data, delta_xp, work_dir):

    # Transform all bounding boxes to compute the target
    all_bboxes = []
    for kind in data.keys():
        for i, target_bbox in enumerate(data[kind]['bboxes']):

            base_shift = data[kind]['deltas'][i]
            shift = base_shift if (kind == 'xpl') else (base_shift + delta_xp)

            warp_matrix = get_translation_matrix(shift)
            translated_bbox = transform_points(target_bbox, warp_matrix)
            all_bboxes.append(translated_bbox)
            debug_log("Image", data[kind]['targets'][i], "to be displaced by:",
                      shift, "from bb", target_bbox, "to bb:", translated_bbox)

    #Intermediate target size encompasses all bounding boxes
    full_bbox = join_bounding_boxes(all_bboxes)
    padding_shift = -full_bbox[0]
    target_size = get_bounding_box_size(full_bbox)
    common_box = intersect_bounding_boxes(all_bboxes) + padding_shift
    debug_log("Common crop box for alignment is", common_box,
              "\nPadding is", padding_shift,
              "\nFull box size is", target_size)

    # Then, align all image files (except the reference) and crop to common box
    manager = mp.Manager()
    partial_result_queue = manager.Queue()
    pool = mp.Pool(processes=4)
    jobs = []
    for kind in data.keys():

        base_shifts = data[kind]['deltas']
        ref_index = data[kind]['ref-index']
        if kind == 'xpl':
            shifts = base_shifts
            ref_align_index = 1 if (ref_index == 0) else (ref_index - 1)
        else:
            shifts = [(delta + delta_xp) for delta in base_shifts]

        for i, (target, delta) in enumerate(zip(data[kind]['targets'], shifts)):

            #put_in_queue = False
            #get_from_queue = False
            #if (kind == 'xpl'):
            #    get_from_queue = (i == ref_index)
            #    put_in_queue = (i == ref_align_index)

            #jobs.append(pool.apply_async(image_align_worker,
            #                            (target, delta, padding_shift,
            #                             target_size, common_box, work_dir,
            #                             partial_result_queue, put_in_queue,
            #                             get_from_queue)))

            jobs.append(pool.apply_async(image_align,
                                        (target, delta, padding_shift,
                                         target_size, common_box, work_dir,
                                         False, None)))
    pool.close()
    pool.join()
    success = all(job.get() for job in jobs)

    return success


def reference_grabber(kind, img_file, crop_size):

    # Get working area (center crop) and bounding box
    img_ref = cvt.file_to_cv2(img_file)
    img_ref_bb = get_bounding_box(img_ref.shape)
    crop_box = get_center_crop_bbox(img_ref, crop_size)
    img_ref_crop = cvt.color_to_gray(crop_to_bounding_box(img_ref, crop_box))

    return (kind, {'crop-ref': img_ref_crop,
                   'bbox-ref': img_ref_bb,
                   'cropbox': crop_box})


def image_list_grabber(kind, img_file, index, crop_box):

    # Get working area and original bounding box
    img = cvt.file_to_cv2(img_file)
    img_crop = cvt.color_to_gray(crop_to_bounding_box(img, crop_box))

    return (kind, index, img_crop, get_bounding_box(img.shape))


def get_registered_averaged(data_dict, crop_size):

    # Generate average intensity grayscale image at cross-polarization
    target_size = (crop_size, crop_size)
    avg = np.zeros(target_size, dtype=float)
    chunk_bbox = get_bounding_box(2*[crop_size])

    converted_bb = []
    for i, xpl_crop in enumerate(data_dict['crops']):
        shift = data_dict['deltas'][i]
        debug_log("getting contrib to avg from", i, "shift", shift)
        warp_matrix = get_translation_matrix(shift)[:2]

        borders = scharr(warp_image(xpl_crop, warp_matrix, target_size))
        borders[borders < np.percentile(borders, 90)] = 0
        avg += borders

        converted_bb.append(transform_points(chunk_bbox, warp_matrix))

    average_stretched = cvt.simple_grayscale_stretch(avg)
    common_box = adjust_bounding_box(chunk_bbox,
                                     intersect_bounding_boxes(converted_bb))
    #averaged_common = crop_to_bounding_box(average_stretched, common_box)
    averaged_common = set_zero_out_of_bounding_box(average_stretched, common_box)

    averaged_bbox = get_bounding_box(averaged_common.shape)
    averaged_shift = np.array(common_box[0][::-1])

    return averaged_common, averaged_bbox, averaged_shift


def get_registered_opaques(data_dict, crop_size):

    # Generate average intensity grayscale image at cross-polarization
    target_size = (crop_size, crop_size)
    opaque = np.ones(target_size, dtype=bool)
    chunk_bbox = get_bounding_box(2*[crop_size])

    converted_bb = []
    for i, xpl_crop in enumerate(data_dict['crops']):
        shift = data_dict['deltas'][i]
        debug_log("getting contrib to opaque from", i, "shift", shift)
        warp_matrix = get_translation_matrix(shift)[:2]

        grays = scharr(warp_image(xpl_crop, warp_matrix, target_size))
        opaques = (grays < 13)
        opaque &= opaques

        converted_bb.append(transform_points(chunk_bbox, warp_matrix))

    common_box = adjust_bounding_box(chunk_bbox,
                                     intersect_bounding_boxes(converted_bb))
    averaged_common = crop_to_bounding_box(opaque, common_box)
    #averaged_common = set_zero_out_of_bounding_box(average_stretched, common_box)

    averaged_bbox = get_bounding_box(opaque.shape)
    averaged_shift = np.array(common_box[0][::-1])

    return averaged_common, averaged_bbox, averaged_shift

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


def pairwise_displacement_worker(index, data_dict, kind):

    crops = data_dict['crops']
    targets = data_dict['targets']

    crop_previous = crops[index-1]
    crop = crops[index]

    peak_loc, peak_val = get_phasecorr_peak(crop_previous, crop, 100)

    debug_log("Translation of", os.path.basename(targets[index]),
              "relative to", os.path.basename(targets[index-1]), "is", peak_loc,
              "value:", peak_val)

    return (kind, index, peak_loc)


def add_pairwise_displacements(pairwise_deltas, reference_index):

    displacements = []

    for i in range(len(pairwise_deltas) + 1):

        if i == reference_index:
            displacements.append(np.r_[0, 0])
            continue

        # displacement[i] = pos[i+1] - pos[i]
        if reference_index < i:
            disp_chain = range(reference_index, i)
            direction = 1
        else:
            disp_chain = range(i, reference_index)
            direction = -1

        total_displacement = sum(pairwise_deltas[j] for j in disp_chain)
        displacements.append(direction * total_displacement)

    return displacements


def image_align_worker(img_file, delta, padding_shift, target_size, crop_box,
                       work_dir, is_reference=False, queue=None,
                       put_in_queue=False):

    ref_align_crop_data = queue.get() if is_reference else None

    success, aligned_center_crop_data = image_align(img_file, delta, padding_shift,
                                                    target_size, crop_box, work_dir,
                                                    is_reference, ref_align_crop_data)
    if put_in_queue:
        queue.put(aligned_center_crop_data)

    return success


def image_align(img_file, delta, padding_shift, target_size,
                crop_box, work_dir, is_reference, correct_shift_ref_crop=None):

    basename = os.path.basename(img_file)
    cf_name = "reg-" + basename
    cropped_file = os.path.join(work_dir, cf_name)

    shift_total = delta + padding_shift[::-1] if not is_reference else delta
    debug_log("Shift to apply (with padding) to", basename + ":", shift_total)
    warp_matrix = get_translation_matrix(shift_total)[:2]
    img = cvt.file_to_cv2(img_file)
    img_pre_aligned = warp_image(img, warp_matrix, target_size)

    if correct_shift_ref_crop is None:
        img_aligned = img_pre_aligned
    else:
        align_corr_bbox, align_corr_crop = correct_shift_ref_crop
        this_crop = cvt.color_to_gray(crop_to_bounding_box(img_pre_aligned,
                                                           align_corr_bbox))
        delta_pair = [align_corr_crop, this_crop]
        delta_align = compute_displacements_direct(delta_pair, 0)[1]
        warp_matrix = get_translation_matrix(delta_align)[:2]
        img_aligned = warp_image(img_pre_aligned, warp_matrix, target_size)

    #cropped = crop_to_bounding_box(img_aligned, crop_box)
    cropped = img_aligned
    success = cvt.cv2_to_file(cropped, cropped_file)

    if success:
        center_crop_box = get_center_crop_bbox(cropped, 1024)
        center_crop = crop_to_bounding_box(cropped, center_crop_box)
        center_crop_name = "crop-" + cf_name.replace(".png", ".jpg")
        center_crop_file = os.path.join(work_dir, center_crop_name)
        cvt.cv2_to_file(center_crop, center_crop_file)

    result = "done" if success else "failed"
    debug_log("Alignment of", img_file, "into", cropped_file, result)

    #center_crop_box = get_center_crop_bbox(cropped, 4096)
    #center_crop = crop_to_bounding_box(cropped, center_crop_box)
    #return success, (center_crop_box, cvt.color_to_gray(center_crop))
    return success


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


def set_zero_out_of_bounding_box(image, bounding_box):

    min_x, max_x = sorted(co[0] for co in bounding_box)
    min_y, max_y = sorted(co[1] for co in bounding_box)

    img_copy = image.copy()
    img_copy[:min_y, :] = 0
    img_copy[max_y::, :] = 0
    img_copy[:, :min_x] = 0
    img_copy[:, max_x:] = 0

    return img_copy


def crop_center_chunk(image, crop_size=1024):

    center = get_bounding_box(image.shape)[1]/2
    center_crop_delta = np.r_[crop_size, crop_size]/2

    corner_start = center - center_crop_delta
    corner_end = center + center_crop_delta

    return crop_to_bounding_box(image, [corner_start, corner_end])


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


