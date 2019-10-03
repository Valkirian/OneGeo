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
    #img1 = cvt.image_load_resize(opt.files[0], opt.reduction)
    img1 = cvt.file_to_cv2(opt.files[0])
    img1_subg = cvt.image_resize(cvt.color_to_gray(img1), opt.reduction)

    # trainImage
    #img2 = cvt.image_load_resize(opt.files[1], opt.reduction)
    img2 = cvt.file_to_cv2(opt.files[1])
    img2_subg = cvt.image_resize(cvt.color_to_gray(img2), opt.reduction)

    # find the keypoints and descriptors
    debug_log("Gather features of images")
    kp1, des1 = sift.detectAndCompute(img1_subg, None)
    kp2, des2 = sift.detectAndCompute(img2_subg, None)

    ref_package = (kp1, des1)
    target_package = (kp2, des2)
    """ Direct
    img2_aligned = match_and_align(ref_package, target_package,
                                   opt.min_matches)
    success = (img2_aligned is not None)
    if success:
        cvt.cv2_to_file(img1, os.path.join(opt.work_dir, basenames[0]))
        cvt.cv2_to_file(img2, os.path.join(opt.work_dir, basenames[1]))
        aligned_file = os.path.join(opt.work_dir, "reg-" + basenames[1])
        cvt.cv2_to_file(img2_aligned, aligned_file)
    """
    warp_matrix = match_get_transf(ref_package, target_package, opt.min_matches)
    print("Warp matrix:")
    print(warp_matrix)
    success = (warp_matrix is not None)

    if success:
        matrix_scaling = get_size_scaling_matrix(opt.reduction)
        scaled_wm = matrix_scaling * warp_matrix
        #scaled_wm = np.dot(matrix_scaling, warp_matrix)
        img2_aligned = align_image(img2, scaled_wm, img1.shape[:2][::-1])

        aligned_file = os.path.join(opt.work_dir, "reg-" + basenames[1])
        cvt.cv2_to_file(img2_aligned, aligned_file)

    result = "done" if success else "failed"
    debug_log("Alignment of", opt.files[1], result)


#def match_and_align(ref_package, target_package, min_matches=10):
def match_get_transf(ref_package, target_package, min_matches=10):

    FLANN_INDEX_KDTREE = 0
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

    warp_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return warp_matrix


def align_image(image, warp_matrix, target_size):

    aligned = cv2.warpPerspective(image, warp_matrix, tuple(target_size),
                                  flags=(cv2.INTER_LINEAR +
                                         cv2.WARP_INVERSE_MAP))
    return aligned


def get_size_scaling_matrix(reduction):

    factor = float(100/reduction)
    scaling = np.array([[1, 1, factor], [1, 1, factor], [factor, factor, 1]])
    #scaling = np.array([[1, 1, factor], [1, 1, factor], [1, 1, 1]])

    return scaling


def process_command_line():

    description = "Finds the optimal common registration among a set of images"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", nargs='*',
            help="Input image files to be registered")
    parser.add_argument('-r', "--reduction", type=int, default=100,
            help="Reduction percentage with which to register images")
    parser.add_argument('-m', "--min-matches", type=int, default=10,
            help="Minimum number of matches to consider a successful registration")
    parser.add_argument('-s', "--sigma", type=float, default=1.6,
            help="Gaussian blur parameter for SIFT")
    parser.add_argument('-c', "--contrast-threshold", type=float, default=0.275,
            help="Contrast threshold parameter for SIFT")
    parser.add_argument('-d', "--work-dir", default='/dev/shm/',
            help="Where to write generated files to")

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
