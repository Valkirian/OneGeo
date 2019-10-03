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
    sift = cv2.xfeatures2d.SIFT_create(**sift_options)

    strip_width = opt.strip_width

    # Reference image
    basename_ref = os.path.basename(opt.reference)
    image_ref = cvt.file_to_cv2(opt.reference)
    imgref_subg = cvt.color_to_gray(image_ref)
    target_size = imgref_subg.shape[::-1]
    imgref_clearance = -1
    kp_ref, des_ref = [], None

    ref_max_clearance = min(target_size) - 2*strip_width

    # Target images
    targets = sorted(set(opt.files) - set([opt.reference,]))
    for img_file in targets:

        basename = os.path.basename(img_file)
        img2 = cvt.file_to_cv2(img_file)
        img2_subg = cvt.color_to_gray(img2)
        keypoints = []
        descriptors = []

        s_clearance = opt.clearance
        tgt_max_clearance = min(img2_subg.shape) - 2*strip_width

        converged = False
        while not converged:

            if s_clearance > imgref_clearance:
                imgref_clearance = s_clearance
                if ref_max_clearance < imgref_clearance:
                    debug_log("Cannot expand feature extraction for reference",
                              basename_ref, "; using current data")
                else:
                    imgref_bands = get_image_bands(imgref_subg, strip_width,
                                                   imgref_clearance)
                    debug_log("Gather features of reference image", basename_ref,
                              "with w =", strip_width, "and c =", imgref_clearance)
                    kp_, des_ = get_features_on_bands(imgref_subg, imgref_bands,
                                                      sift)
                    kp_ref.extend(kp_)
                    des_ref = (des_ if des_ref is None
                               else np.vstack([des_ref, des_]))

                ref_package = (kp_ref, des_ref)

            if tgt_max_clearance < s_clearance:
                debug_log("Cannot expand feature extraction for target",
                            basename_ref, "; Aborting")
                success = False
                break

            debug_log("Gather features of image", basename, "with w =",
                      strip_width, "and c =", s_clearance)
            img2_bands = get_image_bands(img2_subg, strip_width, s_clearance)
            kp_tgt, des_tgt = get_features_on_bands(img2_subg, img2_bands, sift)
            keypoints.extend(kp_tgt)
            descriptors.append(des_tgt)
            target_package = (keypoints, np.vstack(descriptors))

            src_pts_flann, dst_pts_flann = match_points_flann(ref_package, target_package,
                                                  opt.min_matches)
            similarity = get_transf_similarity(src_pts_flann, dst_pts_flann)
            scale_change_pct = 100*abs(similarity.scale - 1)
            success = (scale_change_pct < 1)
            converged = success

            debug_log("Similarity transform: scale change pct.:",
                      scale_change_pct, "Trl:", similarity.translation,
                      "Rot:", similarity.rotation)

            if not success:
                debug_log("Not good enough matching achieved.",
                          "Increasing bands width")
                s_clearance += strip_width
                continue

            transform = SimilarityTransform(scale=1, rotation=similarity.rotation,
                                            translation=similarity.translation)
            warp_matrix = get_transf_homography(src_pts_flann, dst_pts_flann)
            print("Compare Flann matrices:\n  Homography:\n{}\n Similarity:\n{}".format(warp_matrix, transform.params))

            """ BFmatcher test
            src_pts_bf, dst_pts_bf = match_points_bf(ref_package, target_package,
                                                     opt.min_matches)
            warp_matrix_bf = get_transf_homography(src_pts_bf, dst_pts_bf)
            similarity = get_transf_similarity(src_pts_bf, dst_pts_bf, min_samples=2)
            transform_bf = SimilarityTransform(scale=1, rotation=similarity.rotation,
                                               translation=similarity.translation)
            print("Compare BF matrices:\n  Homography:\n{}\n Similarity:\n{}".format(warp_matrix_bf, transform_bf.params))
            """

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
        borders = cvt.simple_grayscale_stretch(scharr(image_band))
        kp, des = detector.detectAndCompute(borders, None)

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

    debug_log( "FLANN matching")
    with Timer(True, True):
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

    debug_log( "FLANN matching")
    with Timer(True, True):
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

    description = "Finds the optimal common registration among a set of images"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("reference", type=str,
            help="Reference image against which the input images will be aligned")
    parser.add_argument("files", nargs='*',
            help="Input image files to be registered")
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
    parser.add_argument('-i', "--small-images", action="store_true",
            help="Writes down-scaled versions of the output images along")

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
