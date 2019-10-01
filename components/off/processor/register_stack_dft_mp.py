#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import argparse
import json
import multiprocessing as mp
import os
import os.path as pth
import sys

import cv2
import numpy as np

from common import (DebugLog, ensure_dir)
import cv_tools as cvt
from dft_stitch import (get_phasecorr_peak,
                        get_translation_matrix, transform_points,
                        get_bounding_box)

debug_log = DebugLog()

#TODO Handling of global reference file with first_is_absolute flag is UNTESTED
def main():

    opt = process_command_line()
    print opt

    ensure_dir(opt.work_dir)

    data = gather_images_data(opt.files, opt.crop_size, opt.threads,
                              opt.use_borders)
    success = align_images(data, opt.work_dir, opt.first_image_is_absolute)

    result = "done" if success else "failed"
    debug_log("Registration job", result)


def gather_images_data(file_list, crop_size, max_threads, use_borders=False,
                       reference_crop_box_corner=None, assume_zero_displacements=False):

    # Pick the first file as the non-moving reference
    ref_file = file_list[0]
    img_ref = cvt.file_to_cv2(ref_file)
    img_ref_bb = get_bounding_box(img_ref.shape)
    # If not given, determine reference crop box corner from the reference file
    img_ref_br = (img_ref_bb[1] if reference_crop_box_corner is None else
                  reference_crop_box_corner)
    crop_box = [(img_ref_br - crop_size)/2, (img_ref_br + crop_size)/2]
    img_crop = crop_to_bounding_box(img_ref, crop_box)
    img_ref_crop = (img_crop if not use_borders else get_borders(img_crop))

    # Read the file lists with few threads as each image is potentially huge
    job_args = ( (img_file, i, crop_box, use_borders) for i, img_file
                in enumerate(file_list[1:]) )
    if max_threads > 1:
        pool = mp.Pool(processes=max_threads)
        jobs = [pool.apply_async(image_list_grabber, args) for args in job_args]
        pool.close()
        pool.join()
        grab_iter = ( job.get() for job in jobs )
    else:
        grab_iter = ( image_list_grabber(*args) for args in job_args )
    data_list = sorted(grab_iter)

    data_dict = {'crops': [img_ref_crop,] + [ elem[1] for elem in data_list ],
                 'bboxes': [img_ref_bb,] + [ elem[2] for elem in data_list ],
                 'targets': file_list}

    if not assume_zero_displacements:
        # Compute displacements w.r.t reference via the image crops just obtained
        job_args = ( (i, data_dict) for i in range(1, len(data_dict['crops'])) )
        if max_threads > 1:
            pool = mp.Pool(processes=max_threads)
            jobs = [ pool.apply_async(displacement_compute_worker, args)
                    for args in job_args ]
            pool.close()
            pool.join()
            disp_iter = ( job.get() for job in jobs )
        else:
            disp_iter = ( displacement_compute_worker(*args) for args in job_args )
        deltas_idx = dict(disp_iter)
        deltas = [delta for i, delta in sorted(deltas_idx.items())]
        deltas_ref = add_pairwise_displacements(deltas, 0)
    else:
        deltas_ref = [ np.r_[0, 0] ]*len(data_dict['crops'])

    data_dict['deltas'] = deltas_ref

    basename_ref = pth.basename(ref_file)
    debug_log("Displacements relative to", basename_ref, deltas_ref)

    return data_dict


def align_images(data_dict, work_dir, first_image_is_absolute=False,
                 max_threads=4, make_center_chunk=True, make_jpeg=False):

    # Transform all bounding boxes to compute the target
    all_bboxes = []
    for i, target_bbox in enumerate(data_dict['bboxes']):

        shift = data_dict['deltas'][i]
        warp_matrix = get_translation_matrix(shift)
        translated_bbox = transform_points(target_bbox, warp_matrix)
        all_bboxes.append(translated_bbox)
        #debug_log("Image", data_dict['targets'][i], "to be displaced by:",
        #            shift, "from bb", target_bbox, "to bb:", translated_bbox)

    if first_image_is_absolute:
        full_bbox = all_bboxes[0]
        padding_shift = np.r_[0, 0]
        common_box = full_bbox
    else:
        #Intermediate target size encompasses all bounding boxes
        full_bbox = join_bounding_boxes(all_bboxes)
        padding_shift = np.maximum([0, 0], -full_bbox[0])
        common_box = intersect_bounding_boxes(all_bboxes) + padding_shift

    target_size = get_bounding_box_size(full_bbox)

    # Finally, align image files and crop to common box
    job_args = []
    job_kwargs = {'make_jpeg': make_jpeg}
    for i, (srcimg, delta) in enumerate(zip(data_dict['targets'],
                                            data_dict['deltas'])):
        if 'tgtpaths' in data_dict:
            target_file = data_dict['tgtpaths'][srcimg]
            target_file_dir = pth.dirname(target_file)
            if not pth.exists(target_file_dir):
                os.makedirs(target_file_dir, 0755)
        else:
            target_file = get_target_filename(srcimg, work_dir)

        if first_image_is_absolute and (delta == np.r_[0, 0]).all():
            # No copy of the reference image is created, but a symlink instead
            os.symlink(srcimg, target_file)

            if make_center_chunk:
                # Small center chunk comes from the reference chunk
                ref_chunk = data_dict['crops'][i]
                gen_center_small_chunk(ref_chunk, target_file, 1024)
        else:
            args = (srcimg, target_file, delta, padding_shift, target_size,
                    common_box, make_center_chunk)
            job_args.append(args)

    if max_threads > 1:
        pool = mp.Pool(processes=max_threads)
        jobs = [ pool.apply_async(image_align_worker, args, job_kwargs) for args
                in job_args ]
        pool.close()
        pool.join()
        result_iter = ( job.get() for job in jobs )
    else:
        result_iter = ( image_align_worker(*args, **job_kwargs) for args in job_args )
    success = all(result_iter)

    return success


def image_list_grabber(img_file, index, crop_box, use_borders=False):

    img = cvt.file_to_cv2(img_file)

    img_crop = crop_to_bounding_box(img, crop_box)
    img_crop_tgt = (img_crop if not use_borders else
                    get_borders(img_crop))

    return (index, img_crop_tgt, get_bounding_box(img.shape))


def displacement_compute_worker(index, data_dict):

    crops = data_dict['crops']
    targets = data_dict['targets']

    crop_ref = cvt.color_to_gray(crops[index-1])
    crop = cvt.color_to_gray(crops[index])

    peak_loc, peak_val = get_phasecorr_peak(crop_ref, crop, 100)

    print ("Translation of", pth.basename(targets[index]),
           "relative to", pth.basename(targets[index-1]), "is",
           peak_loc, "value:", peak_val)

    return (index, peak_loc)


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

        total_displacement = direction * sum(pairwise_deltas[j] for j in disp_chain)
        displacements.append(total_displacement)
        print ("Displacement from", i, "to", reference_index, "is",
               total_displacement)

    return displacements


def image_align_worker(img_file, target_file, delta, padding_shift, target_size,
                       crop_box, make_center_chunk, make_jpeg=False):

    img = cvt.file_to_cv2(img_file)

    total_shift = delta + padding_shift[::-1]
    cropped = translate_crop(img, total_shift, crop_box, target_size)
    success = cvt.cv2_to_file(cropped, target_file)

    if success:
        registration_file = pth.splitext(target_file)[0] + ".json"
        registration_save(total_shift, crop_box, target_size, registration_file)

    if success and make_center_chunk:
        gen_center_small_chunk(cropped, target_file, 1024)

    if success and make_jpeg:
        reduced_fpath = pth.splitext(target_file)[0] + '.jpg'
        cvt.cv2_to_file(cropped, reduced_fpath)

    result = "done" if success else "failed"
    debug_log("Alignment of", img_file, "into", target_file, result)

    return success


def translate_crop(image, shift, crop_box, target_size):

    warp_matrix = get_translation_matrix(shift)[:2]
    img_aligned = warp_image(image, warp_matrix, target_size)
    cropped = crop_to_bounding_box(img_aligned, crop_box)

    return cropped


def get_target_filename(source_image_filename, work_dir):

    basename = pth.basename(source_image_filename)
    cf_name = "reg-" + basename
    return pth.join(work_dir, cf_name)


def gen_center_small_chunk(image, image_filename, size):

    img_path, img_fname = pth.split(image_filename)
    center_crop = crop_center_chunk(image, size)
    center_crop_name = "small-crop-" + img_fname.replace(".png", ".jpg")
    center_crop_file = pth.join(img_path, center_crop_name)

    return cvt.cv2_to_file(center_crop, center_crop_file)


def warp_image(image, warp_matrix, target_size):

    size_x, size_y = target_size
    aligned = cv2.warpAffine(image, warp_matrix, (size_x, size_y),
                             flags=(cv2.INTER_LINEAR))
    return aligned


def get_borders(rgb_image, l_thd=100, u_thd=300):

    img_can = cv2.merge([cv2.Canny(chan, l_thd, u_thd) for chan
                         in cv2.split(rgb_image)])
    img_can_value = img_can.max(axis=2)

    return img_can_value


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


def registration_save(shift, crop_box, target_size, target_file):
    
    to_save = {'shift': list(shift), 'crop_box': crop_box.tolist(),
               'size': target_size.tolist()}
    with open(target_file, 'w') as fobj:
        json.dump(to_save, fobj)


def registration_load(source_file):

    with open(source_file) as fobj:
        data = json.load(fobj)

    data_np = { key: np.array(arr) for key, arr in data.items() }
    return data_np


def process_command_line():

    description = ("Finds the optimal common registration among a set of images"
                   "by computing relative translations")
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", nargs='*',
            help="Input image files to be registered")
    parser.add_argument('-c', "--crop-size", type=int, default=4096,
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

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())


