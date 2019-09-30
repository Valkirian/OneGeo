#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys

import cv2
import numpy as np

from common import (DebugLog, ensure_dir)
import cv_tools as cvt
from register_stack_sift import (get_image_bands, get_features_on_bands,
                                 match_get_transf, is_good_matrix,
                                 transform_points, get_bounding_box,
                                 crop_center_chunk)

debug_log = DebugLog()

def main():

    opt = process_command_line()
    print opt

    ensure_dir(opt.work_dir)

    sift_options = dict(sigma=opt.sigma,
                        contrastThreshold=opt.contrast_threshold)
    detector = cv2.xfeatures2d.SIFT_create(**sift_options)

    good_mat, mat = get_registration(opt.ppl_crops, opt.xpl_crops, detector,
                                     opt.strip_width, opt.clearance,
                                     opt.min_matches)

    if not good_mat:
        debug_log("Unable to find a proper transformation")
        return 1

    debug_log("PPL->XPL transform is", mat)
    align_images(opt.work_dir, opt.ppl_files, opt.xpl_files, mat)

    return 0


def get_registration(ppl_crops, xpl_crops, detector, strip_width, clearance,
                     min_matches):

    debug_log("Reading XPL crops" )
    xpl_avg = get_rgb_average(xpl_crops)
    debug_log("Reading PPL crops" )
    ppl_avg = get_rgb_average(ppl_crops)

    debug_log("Matching crops")
    success, warp_matrix = compute_registration(xpl_avg, ppl_avg, detector,
                                                strip_width, clearance,
                                                min_matches)

    return success, warp_matrix


def align_images(work_dir, ppl_files, xpl_files, warp_matrix):

    # Get reference bounding box
    for xpl_file in xpl_files:
        metadata_json = xpl_file.replace(".png", ".metadata.json")
        if os.path.isfile(metadata_json):
            xpl_h, xpl_w = json.load(open(metadata_json))
        else:
            xpl_image = cvt.file_to_cv2(xpl_file)
            xpl_h, xpl_w = xpl_image.shape[:2]
        break

    target_size = (xpl_w, xpl_h)


    # Finally, align image files and crop to common box
    for ppl_file in ppl_files:

        img = cvt.file_to_cv2(ppl_file)
        img_aligned = warp_image(img, warp_matrix, target_size)
        img_bbox = get_bounding_box(img.shape)
        new_bbox = transform_points(img_bbox, warp_matrix)

        print "Start bbox:", img_bbox[0], img_bbox[-1]
        print "New bbox:", new_bbox

        #cropped = crop_to_bounding_box(img_aligned, common_box)
        cropped = img_aligned

        basename = os.path.basename(ppl_file)
        cf_name = "reg-" + basename
        cropped_file = os.path.join(work_dir, cf_name)
        success = cvt.cv2_to_file(cropped, cropped_file)

        if success:
            center_crop = crop_center_chunk(cropped, 1024)
            center_crop_name = "crop-" + cf_name.replace(".png", ".jpg")
            center_crop_file = os.path.join(work_dir, center_crop_name)
            cvt.cv2_to_file(center_crop, center_crop_file)

        result = "done" if success else "failed"
        debug_log("Alignment of", ppl_file, "into", cropped_file,result)


def get_rgb_average(images_file_list):

    avg = cvt.file_to_cv2(os.path.expanduser(images_file_list[0])).astype(float)
    for img_file in images_file_list[1:]:
        avg += cvt.file_to_cv2(os.path.expanduser(img_file))
    avg /= len(images_file_list)

    return avg.astype(np.uint8)


def compute_registration(img_reference, img_target, detector, strip_width,
                         clearance, min_matches=10):

    imgref_subg = cvt.color_to_gray(img_reference)
    target_size = imgref_subg.shape[::-1]
    imgref_clearance = -1
    kp_ref, des_ref = [], None

    ref_max_clearance = min(target_size) - 2*strip_width

    img2_subg = cvt.color_to_gray(img_target)
    keypoints = []
    descriptors = []

    s_clearance = clearance
    tgt_max_clearance = min(img2_subg.shape) - 2*strip_width

    converged = False
    while not converged:

        if s_clearance > imgref_clearance:
            imgref_clearance = s_clearance
            if ref_max_clearance < imgref_clearance:
                debug_log("Cannot expand feature extraction for reference (XPL)"
                          "; using current data")
            else:
                imgref_bands = get_image_bands(imgref_subg, strip_width,
                                                imgref_clearance)
                debug_log("Gather features of reference image",
                          "with w =", strip_width, "and c =", imgref_clearance)
                kp_, des_ = get_features_on_bands(imgref_subg, imgref_bands,
                                                    detector)
                kp_ref.extend(kp_)
                des_ref = (des_ if des_ref is None
                            else np.vstack([des_ref, des_]))

            ref_package = (kp_ref, des_ref)

        if tgt_max_clearance < s_clearance:
            debug_log("Cannot expand feature extraction for target; Aborting")
            success = False
            break

        debug_log("Gather features of target image with w =",
                    strip_width, "and c =", s_clearance)
        img2_bands = get_image_bands(img2_subg, strip_width, s_clearance)
        kp_tgt, des_tgt = get_features_on_bands(img2_subg, img2_bands, detector)
        keypoints.extend(kp_tgt)
        descriptors.append(des_tgt)
        target_package = (keypoints, np.vstack(descriptors))

        warp_matrix = match_get_transf(ref_package, target_package, min_matches)
        success = (warp_matrix is not None and
                    is_good_matrix(warp_matrix, 1e-7))
        converged = success

        if not success:
            debug_log("Not good enough matching achieved.",
                        "Increasing bands width")
            s_clearance += strip_width
            continue

    return success, warp_matrix


def warp_image(image, warp_matrix, target_size):

    #aligned = cv2.warpPerspective(image, warp_matrix, tuple(target_size),
    aligned = cv2.warpAffine(image, warp_matrix[:2], tuple(target_size),
                                  flags=(cv2.INTER_LINEAR +
                                         cv2.WARP_INVERSE_MAP))

    return aligned


def process_command_line():

    description = ("Finds the optimal common registration among a set of images"
                   "by computing relative translations")
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--ppl-files", nargs='+',
            help="Plane-polarized full-size images to be registered")
    parser.add_argument("--ppl-crops", nargs='+',
            help="Plane-polarized center-based cropped images to use as "
                 "registration targets")
    parser.add_argument("--xpl-files", nargs='+',
            help="Cross-polarized full-size images to be registered")
    parser.add_argument("--xpl-crops", nargs='+',
            help="Cross-polarized center-based cropped images to use as "
                 "registration references")
    parser.add_argument('-m', "--min-matches", type=int, default=20,
            help="Minimum number of matches to consider a successful registration")
    parser.add_argument('-w', "--strip-width", type=int, default=1000,
            help="image bands width to use for feature extraction")
    parser.add_argument('-c', "--clearance", type=int, default=1000,
            help=("Initial image bands clearance off borders to use "
                  "for feature extraction"))
    parser.add_argument('-s', "--sigma", type=float, default=1.6,
            help="Gaussian blur parameter for SIFT")
    parser.add_argument('-t', "--contrast-threshold", type=float, default=0.1,
            help="Contrast threshold parameter for SIFT")
    parser.add_argument('-d', "--work-dir", default='/dev/shm/',
            help="Where to write generated files to")
    parser.add_argument('-k', "--keep-uncropped", action="store_true",
            help="Avoids erasing of aligned uncropped output images")

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
