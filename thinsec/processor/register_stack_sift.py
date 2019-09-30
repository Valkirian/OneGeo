#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import argparse
import os
import sys

import cv2
import numpy as np
from skimage.transform import SimilarityTransform
from skimage.measure import ransac
from skimage.filters import scharr

from common import (DebugLog, ensure_dir)
import cv_tools as cvt
from timer import Timer

debug_log = DebugLog()


def main():

    opt = process_command_line()
    print opt

    ensure_dir(opt.work_dir)

    # Feature detector
    sift_options = dict(sigma=opt.sigma,
                        contrastThreshold=opt.contrast_threshold)

    base_csize = opt.crop_size

    # Reference image
    basename_ref = os.path.basename(opt.reference)
    image_ref = cvt.file_to_cv2(opt.reference)
    imgref_subg = cvt.color_to_gray(image_ref)
    target_size = imgref_subg.shape[::-1]
    imgref_csize = -1
    kp_ref, des_ref = [], None

    ref_max_size = min(target_size)

    # Target images
    targets = sorted(set(opt.files) - set([opt.reference,]))
    for img_file in targets:

        basename = os.path.basename(img_file)
        img2 = cvt.file_to_cv2(img_file)
        img2_subg = cvt.color_to_gray(img2)
        keypoints = []
        descriptors = []

        csize = base_csize
        tgt_max_size = min(img2_subg.shape)

        converged = False
        while not converged:

            if csize > imgref_csize:
                imgref_csize = csize
                if ref_max_size < imgref_csize:
                    debug_log("Cannot expand feature extraction for reference",
                              basename_ref, "; using current data")
                else:
                    debug_log("Gather features of reference image", basename_ref,
                              "with c =", imgref_csize)
                    imgref_bands = get_center_crop_bounding_box(imgref_subg,
                                                                imgref_csize)
                    print("Crop box:", imgref_bands)
                    sift = cv2.xfeatures2d.SIFT_create(**sift_options)
                    roi = crop_to_bounding_box(imgref_subg, imgref_bands)
                    #borders = cvt.simple_grayscale_stretch(scharr(roi))
                    kp_ref, des_ref = sift.detectAndCompute(roi, None)

                ref_package = (kp_ref, des_ref)

            if tgt_max_size < csize:
                debug_log("Cannot expand feature extraction for target",
                          basename_ref, "; Aborting")
                success = False
                break

            debug_log("Gather features of image", basename, "with c =", csize)
            sift = cv2.xfeatures2d.SIFT_create(**sift_options)
            roi = crop_to_bounding_box(img2_subg, imgref_bands)
            #borders = cvt.simple_grayscale_stretch(scharr(roi))
            keypoints, descriptors = sift.detectAndCompute(roi, None)
            target_package = (keypoints, descriptors)

            src_pts_flann, dst_pts_flann = match_points_flann(ref_package, target_package,
                                                  opt.min_matches)
            similarity = get_transf_similarity(src_pts_flann, dst_pts_flann)
            scale_change_pct = 100*abs(similarity.scale - 1)

            debug_log("FLANN Similarity transform: scale change pct.:",
                      scale_change_pct, "Trl:", similarity.translation,
                      "Rot:", similarity.rotation)
            warp_matrix = get_transf_homography(src_pts_flann, dst_pts_flann)
            print("Compare FLANN matrices:\n  Homography:\n{}\n Similarity:\n{}".format(warp_matrix, similarity.params))

            """ BFmatcher test
            src_pts_bf, dst_pts_bf = match_points_bf(ref_package, target_package,
                                                     opt.min_matches)
            similarity = get_transf_similarity(src_pts_bf, dst_pts_bf, min_samples=2)
            scale_change_pct = 100*abs(similarity.scale - 1)

            debug_log("BF Similarity transform: scale change pct.:",
                      scale_change_pct, "Trl:", similarity.translation,
                      "Rot:", similarity.rotation)

            transform = SimilarityTransform(scale=1, rotation=similarity.rotation,
                                            translation=similarity.translation)
            warp_matrix_bf = get_transf_homography(src_pts_bf, dst_pts_bf)
            print("Compare BF matrices:\n  Homography:\n{}\n  Similarity:\n{}".format(warp_matrix_bf, transform.params))
            """

            success = (scale_change_pct < 1)
            converged = success

            if not success:
                debug_log("Not good enough matching achieved.",
                          "Increasing bands width")
                csize += base_csize
                continue

            transform = SimilarityTransform(scale=1, rotation=similarity.rotation,
                                            translation=similarity.translation)

            img2_aligned = align_image(img2, transform.params, target_size)

            aligned_file = os.path.join(opt.work_dir, "reg-" + basename)
            success = cvt.cv2_to_file(img2_aligned, aligned_file)

            if success:
                center_crop = crop_center_chunk(img2_aligned, 1024)
                center_crop_name = "crop-" + basename.replace(".png", ".jpg")
                center_crop_file = os.path.join(opt.work_dir, center_crop_name)
                cvt.cv2_to_file(center_crop, center_crop_file)

                if opt.small_images:
                    small = cvt.image_resize(img2_aligned, 30)
                    small_name = "small-" + basename.replace(".png", ".jpg")
                    small_file = os.path.join(opt.work_dir, small_name)
                    cvt.cv2_to_file(small, small_file)

        result = "done" if success else "failed"
        debug_log("Alignment of", img_file, result)


def get_image_bands(image_gray, strip_width=1200, border_clearance=1000):

    h, w = image_gray.shape

    # starting corners
    to_ba_corner = np.r_[border_clearance, border_clearance]
    ri_ba_corner = np.r_[border_clearance, w - (border_clearance + strip_width)]
    bt_ba_corner = np.r_[h - (border_clearance + strip_width),
                         border_clearance + strip_width]
    le_ba_corner = np.r_[border_clearance + strip_width, border_clearance]

    # band sizes
    ho_bands_sz = np.r_[strip_width, w - (2*border_clearance + strip_width)]
    ve_bands_sz = np.r_[h - (2*border_clearance + strip_width), strip_width]

    # bounding corners
    top_band_sc = [to_ba_corner, to_ba_corner + ho_bands_sz]
    right_band_sc = [ri_ba_corner, ri_ba_corner + ve_bands_sz]
    bot_band_sc = [bt_ba_corner, bt_ba_corner + ho_bands_sz]
    left_band_sc = [le_ba_corner, le_ba_corner + ve_bands_sz]

    debug_log("Band corners:", (to_ba_corner, ri_ba_corner,
                                bt_ba_corner, le_ba_corner))
    return (top_band_sc, right_band_sc, bot_band_sc, left_band_sc)


def get_features_on_bands(source_image_gray, bands, detector):

    keypoints = []
    descriptors = []

    for i_band, band in enumerate(bands):
        corner_start, corner_end = band

        image_band = source_image_gray[corner_start[0]:corner_end[0],
                                       corner_start[1]:corner_end[1]]
        kp, des = detector.detectAndCompute(image_band, None)

        if des is None:
            return None, None
        descriptors.append(des)

        for keyp in kp:
            new_kp = cv2.KeyPoint(x=keyp.pt[0] + corner_start[1],
                                  y=keyp.pt[1] + corner_start[0],
                                  _size=keyp.size, _angle=keyp.angle,
                                  _response=keyp.response, _octave=keyp.octave,
                                  _class_id=keyp.class_id)
            keypoints.append(new_kp)

    return keypoints, np.vstack(descriptors)


def match_points_flann(ref_package, target_package, min_matches=10):

    kp_ref, des_ref = ref_package
    kp_tgt, des_tgt = target_package

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
    search_params = dict(checks = 50)

    with Timer(True, True, "FLANN matching"):
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des_ref, des_tgt, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = [ m for m, n in matches if m.distance < 0.7*n.distance ]

    if len(good) < min_matches:
        print "Not enough matches are found - %d/%d" % (len(good), min_matches)
        return None, None

    print "Good matches: {}/{}".format(len(good), len(matches))
    src_pts = np.float64([ kp_ref[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float64([ kp_tgt[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    return src_pts, dst_pts


def match_points_bf(ref_package, target_package, min_matches=10):

    kp_ref, des_ref = ref_package
    kp_tgt, des_tgt = target_package

    with Timer(True, True, "BF matching"):
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.knnMatch(des_ref, des_tgt, k=1)

    # As per OpenCV's docs, a BFMatcher with crossCheck=True and knnMatch
    #  invoked with k=1 returns only consistent matches
    good = [ m[0] for m in matches if m ]

    if len(good) < min_matches:
        print "Not enough matches are found - %d/%d" % (len(good), min_matches)
        return None, None

    print "Good matches: {}/{}".format(len(good), len(matches))
    src_pts = np.float64([ kp_ref[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float64([ kp_tgt[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    return src_pts, dst_pts


def get_transf_homography(src_pts, dst_pts):

    warp_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 6.0)

    return warp_matrix


def get_transf_similarity(src_pts, dst_pts, min_samples=5, max_trials=2000,
                          residual_threshold=1):

    src = src_pts.squeeze()
    dst = dst_pts.squeeze()

    # robustly estimate affine transform model with RANSAC
    model_robust, inliers = ransac((src, dst), SimilarityTransform,
                                   min_samples=min_samples,
                                   residual_threshold=residual_threshold,
                                   max_trials=max_trials)
    #outliers = inliers == False

    return model_robust


def is_good_matrix(warp_matrix, tol=1e-7):

    # We expect pure translation and rotations, so perspective deformation
    # coefficients must be vanishingly small
    perspective = warp_matrix[2, :2]
    divergence = np.abs(perspective).max()

    print("Matrix divergence: {:.4g}".format(divergence))
    return (divergence < tol)


def align_image(image, warp_matrix, target_size):

    aligned = cv2.warpAffine(image, warp_matrix[:2], tuple(target_size),
                             flags=(cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP))
    return aligned


def get_adjusted_bounding_box(ref_bb, target_bb, warp_matrix):

    target_warped_bb = transform_points(target_bb, warp_matrix)
    adjusted_bb = adjust_bounding_box(ref_bb, target_warped_bb)

    return adjusted_bb


def transform_points(points, M):

    transformed = []
    m = np.linalg.inv(M)

    for p in points:
        x, y = p
        x_t = (m[0,0]*x + m[0,1]*y + m[0,2])/(m[2,0]*x + m[2,1]*y + m[2,2])
        y_t = (m[1,0]*x + m[1,1]*y + m[1,2])/(m[2,0]*x + m[2,1]*y + m[2,2])

        transformed.append([x_t, y_t])

    return np.array(transformed).round().astype(int)


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

    x_coords = [co[0] for co in bounding_box]
    y_coords = [co[1] for co in bounding_box]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    im_roi = image[min_y:max_y, min_x:max_x]

    return im_roi


def get_center_crop_bounding_box(image, chunk_length=1024):

    center = np.array(image.shape[:2][::-1])/2
    center_crop_delta = np.r_[chunk_length, chunk_length]/2

    corner_start = center - center_crop_delta
    corner_end = center + center_crop_delta

    return (corner_start, corner_end)


def crop_center_chunk(image, chunk_length=1024):

    center = np.array(image.shape[:2])/2
    center_crop_delta = np.r_[chunk_length, chunk_length]/2
    corner_start = center - center_crop_delta
    corner_end = center + center_crop_delta
    center_crop = image[corner_start[0]:corner_end[0],
                        corner_start[1]:corner_end[1]]
    return center_crop


def process_command_line():

    description = "Finds the optimal common registration among a set of images"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("reference", type=str,
            help="Reference image against which the input images will be aligned")
    parser.add_argument("files", nargs='*',
            help="Input image files to be registered")
    parser.add_argument('-m', "--min-matches", type=int, default=20,
            help="Minimum number of matches to consider a successful registration")
    parser.add_argument('-c', "--crop-size", type=int, default=2048,
            help="Center-crop box length to use for feature extraction")
    parser.add_argument('-s', "--sigma", type=float, default=1.6,
            help="Gaussian blur parameter for SIFT")
    parser.add_argument('-t', "--contrast-threshold", type=float, default=0.1,
            help="Contrast threshold parameter for SIFT")
    parser.add_argument('-d', "--work-dir", default='/dev/shm/',
            help="Where to write generated files to")
    parser.add_argument('-k', "--keep-uncropped", action="store_true",
            help="Avoids erasing of aligned uncropped output images")
    parser.add_argument('-i', "--small-images", action="store_true",
            help="Writes down-scaled versions of the output images along")

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
