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

from timer import Timer
import cv_tools as cvt
from common import ensure_dir


def main():

    opt = process_command_line()
    print opt

    ensure_dir(opt.work_dir)
    basenames = [os.path.splitext(os.path.basename(fi))[0] for fi in opt.files]

    # queryImage
    img1 = cvt.color_to_gray(cvt.image_load_resize(opt.files[0], opt.reduction))
    #img1_wim = cvt.simple_grayscale_stretch(scharr(img1))
    #img1_wim = cv2.Canny(img1, 100, 200)
    img1_wim = img1

    # trainImage
    img2 = cvt.color_to_gray(cvt.image_load_resize(opt.files[1], opt.reduction))
    #img2_wim = cvt.simple_grayscale_stretch(scharr(img2))
    #img2_wim = cv2.Canny(img2, 100, 200)
    img2_wim = img2

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=opt.sift_contrast_thd,
                                       sigma=opt.sift_sigma)

    # find the keypoints and descriptors with SIFT
    with Timer(True, True, "Get SIFT for images"):
        kp1, des1 = sift.detectAndCompute(img1_wim, None)
        kp2, des2 = sift.detectAndCompute(img2_wim, None)

    file_keyp = os.path.join(opt.work_dir, "{}-siftkeys.jpg".format(basenames[0]))
    draw_keypoints(img1_wim, kp1, 3, (0, 255, 0), file_keyp)
    print "Image 1 has {} keypoints".format(len(kp1))

    file_keyp = os.path.join(opt.work_dir, "{}-siftkeys.jpg".format(basenames[1]))
    draw_keypoints(img2_wim, kp2, 3, (0, 255, 0), file_keyp)
    print "Image 2 has {} keypoints".format(len(kp2))

    src_pts, dst_pts, good = match_points_bf((kp1, des1), (kp2, des2))
    #src_pts, dst_pts, good = match_points_flann((kp1, des1), (kp2, des2))

    M, mask = get_transf_homography(src_pts, dst_pts)
    print "Homography matrix:\n", M
    trf, mask = get_transf_similarity(src_pts, dst_pts)

    print "Similarity matrix:\n", trf.params
    print "Similarity params: R({}), S({}), T({})".format(trf.rotation,
                                                          trf.scale,
                                                          trf.translation)

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, trf.params)
    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    matchesMask = mask.ravel().astype(bool).tolist()
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    out_file = os.path.join(opt.work_dir, "match_{}-{}.jpg".format(*basenames))
    cv2.imwrite(out_file, img3)

    return 0


def draw_keypoints(image, keypoints, radius, color, out_file):

    img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for k in keypoints:
        cv2.circle(img, tuple(map(int, map(round, k.pt))), radius, color, -1)

    cv2.imwrite(out_file, img)


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

    return src_pts, dst_pts, good


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

    return src_pts, dst_pts, good


def get_transf_homography(src_pts, dst_pts):

    warp_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 6.0)

    return warp_matrix, mask


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

    return model_robust, inliers


def process_command_line():

    description = "Finds the optimal common registration among a set of images"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", nargs='*',
            help="Input image files to be registered")
    parser.add_argument('-r', "--reduction", type=int, default=100,
            help="Reduction percentage with which to register images")
    parser.add_argument('-m', "--min-matches", type=int, default=10,
            help="Minimum number of matches to consider a successful registration")
    parser.add_argument('-s', "--sift-sigma", type=float, default=1.6,
            help="Gaussian blur parameter for SIFT")
    parser.add_argument('-t', "--sift-contrast-thd", type=float, default=0.1,
            help="Contrast Threshold parameter for SIFT")
    parser.add_argument('-d', "--work-dir", default='/dev/shm/',
            help="Where to write generated files to")

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
