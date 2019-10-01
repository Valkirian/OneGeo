#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import argparse
import os
import sys

import cv2
import numpy as np

from common import DebugLog
import cv_tools as cvt
from LibStitch import ensure_dir

debug_log = DebugLog()


def main():

    opt = process_command_line()
    print opt

    ensure_dir(opt.work_dir)
    basenames = [os.path.basename(fi) for fi in opt.files]

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=opt.contrast_threshold,
                                       sigma=opt.sigma)

    # queryImage
    img1_subg = cvt.color_to_gray(cvt.file_to_cv2(opt.files[0]))
    target_size = img1_subg.shape[::-1]

    # trainImage
    img2 = cvt.file_to_cv2(opt.files[1])
    img2_subg = cvt.color_to_gray(img2)

    """
    debug_render_bands(img1_subg,
                       os.path.join(opt.work_dir, "banded-" + basenames[0]),
                       strip_width=2000)
    debug_render_bands(img2_subg,
                       os.path.join(opt.work_dir, "banded-" + basenames[1]),
                       strip_width=2000)
    """

    img1_bands = get_image_bands(img1_subg, opt.strip_width)
    img2_bands = get_image_bands(img2_subg, opt.strip_width)

    # find the keypoints and descriptors
    debug_log("Gather features of images")
    #kp1, des1 = get_features_on_bands(img1_subg, img1_bands, sift, basenames[0])
    kp1, des1 = get_features_on_bands(img1_subg, img1_bands, sift)
    kp2, des2 = get_features_on_bands(img2_subg, img2_bands, sift)

    """ Debug keypoint shifting in bands
    file_out = "/dev/shm/{}-banded.jpg".format(basenames[0])
    draw_keypoints(img1_subg, kp1, 80, (0, 255, 0), file_out)
    """

    ref_package = (kp1, des1)
    target_package = (kp2, des2)

    warp_matrix = match_get_transf(ref_package, target_package, opt.min_matches)
    print("Warp matrix:")
    print(warp_matrix)
    success = (warp_matrix is not None)

    if success:
        img2_aligned = align_image(img2, warp_matrix, target_size)
        aligned_file = os.path.join(opt.work_dir, "reg-" + basenames[1])
        cvt.cv2_to_file(img2_aligned, aligned_file)

    result = "done" if success else "failed"
    debug_log("Alignment of", opt.files[1], result)


def get_image_bands(image_gray, strip_width=1200, border_clearance=1000,
                    debug_offset=0):

    h, w = image_gray.shape
    of = debug_offset

    # starting corners
    to_ba_corner = [border_clearance, border_clearance]
    ri_ba_corner = [border_clearance, w - (border_clearance + strip_width)]
    bt_ba_corner = [h - (border_clearance + strip_width), border_clearance + strip_width]
    le_ba_corner = [border_clearance + strip_width, border_clearance]

    # band sizes
    ho_bands_sz = [strip_width, w - (2*border_clearance + strip_width)]
    ve_bands_sz = [h - (2*border_clearance + strip_width), strip_width]

    # slice ranges
    top_band_sc = [[co, co + sz - of] for co, sz in zip(to_ba_corner, ho_bands_sz)]
    right_band_sc = [[co, co + sz - of] for co, sz in zip(ri_ba_corner, ve_bands_sz)]
    bottom_band_sc = [[co, co + sz - of] for co, sz in zip(bt_ba_corner, ho_bands_sz)]
    left_band_sc = [[co, co + sz - of] for co, sz in zip(le_ba_corner, ve_bands_sz)]

    return (top_band_sc, right_band_sc, bottom_band_sc, left_band_sc)


def debug_render_bands(source_image_gray, filename_out, strip_width=1200,
                       border_clearance=1000):

    bands = get_image_bands(source_image_gray, strip_width, border_clearance,
                            debug_offset=10)
    banded_test = np.zeros_like(source_image_gray)
    for band in bands:
        banded_test[band[0][0]:band[0][1], band[1][0]:band[1][1]] = (
                source_image_gray[band[0][0]:band[0][1], band[1][0]:band[1][1]])
    cvt.cv2_to_file(banded_test, filename_out)


def get_features_on_bands(source_image_gray, bands, detector, name=None):

    keypoints = []
    descriptors = []

    for i, band in enumerate(bands):
        image_band = source_image_gray[band[0][0]:band[0][1],
                                       band[1][0]:band[1][1]]
        kp, des = detector.detectAndCompute(image_band, None)
        descriptors.append(des)
        for keyp in kp:
            new_kp = cv2.KeyPoint(x=keyp.pt[0] + band[1][0],
                                  y=keyp.pt[1] + band[0][0],
                                  _size=keyp.size, _angle=keyp.angle,
                                  _response=keyp.response, _octave=keyp.octave,
                                  _class_id=keyp.class_id)
            keypoints.append(new_kp)

        if name is not None:
            band_file_out = "/dev/shm/{}-band{}.jpg".format(name, i)
            draw_keypoints(image_band, kp, 40, (0, 255, 0), band_file_out)

    return keypoints, np.vstack(descriptors)


def match_get_transf(ref_package, target_package, min_matches=10):

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
    search_params = dict(checks = 50)

    kp1, des1 = ref_package
    kp2, des2 = target_package

    debug_log("FLANN matching")
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = [ m for m, n in matches if m.distance < 0.7*n.distance ]
    debug_log("There are", len(good), "good matches")

    if len(good) < min_matches:
        debug_log("Not enough matches are found ({}/{})".format(len(good),
                                                                min_matches))
        return None

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    warp_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

    return warp_matrix


def align_image(image, warp_matrix, target_size):

    aligned = cv2.warpPerspective(image, warp_matrix, tuple(target_size),
                                  flags=(cv2.INTER_LINEAR +
                                         cv2.WARP_INVERSE_MAP))
    return aligned


def draw_keypoints(image, keypoints, radius, color, out_file):

    img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for k in keypoints:
        cv2.circle(img, tuple(map(int, map(round, k.pt))), radius, color, -1)

    cv2.imwrite(out_file, img)


def process_command_line():

    description = "Finds the optimal common registration among a set of images"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", nargs='*',
            help="Input image files to be registered")
    parser.add_argument('-m', "--min-matches", type=int, default=10,
            help="Minimum number of matches to consider a successful registration")
    parser.add_argument('-w', "--strip-width", type=int, default=1200,
            help="image bands width to use for feature extraction")
    parser.add_argument('-s', "--sigma", type=float, default=1.6,
            help="Gaussian blur parameter for SIFT")
    parser.add_argument('-c', "--contrast-threshold", type=float, default=0.275,
            help="Contrast threshold parameter for SIFT")
    parser.add_argument('-d', "--work-dir", default='/dev/shm/',
            help="Where to write generated files to")

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
