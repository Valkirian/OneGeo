#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import argparse
import os

import cv2
import numpy as np

import cv_tools as cvt
from common import DebugLog

debug_log = DebugLog()


def main():

    opt = process_command_line()
    print opt

    targets = sorted(set(opt.files) - set([opt.reference,]))

    image0 = cvt.file_to_cv2(opt.reference)

    for image_fi in targets:
        basename = os.path.basename(image_fi)
        debug_log("Matching", os.path.basename(opt.reference), "with", basename)
        image = cvt.file_to_cv2(image_fi)
        output_file = os.path.join(opt.work_dir, "pre-" + basename)

        register_image_pair(image0, image, opt.reduction, output_file)


def register_image_pair(im1, im2, roi_size, file_out):

    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    im1_roi = get_center_roi(im1_gray, roi_size)
    im2_roi = get_center_roi(im2_gray, roi_size)

    # Find size of image1
    sz = im1.shape

    # Define the motion model
    warp_mode = cv2.MOTION_EUCLIDEAN
    #warp_mode = cv2.MOTION_AFFINE

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000;

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-9;

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(im1_roi, im2_roi, warp_matrix,
                                             warp_mode, criteria)

    print("Warp matrix ref->{}:".format(os.path.basename(file_out)))
    print(warp_matrix)

    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1], sz[0]),
                                          flags=(cv2.INTER_LINEAR +
                                                 cv2.WARP_INVERSE_MAP))
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]),
                                     flags=(cv2.INTER_LINEAR +
                                            cv2.WARP_INVERSE_MAP));

    success = cv2.imwrite(file_out, im2_aligned)

    return success


def get_center_roi(image, size):

    h, w = image.shape[:2]
    roi = image[(h-size)/2:(h+size)/2, (w-size)/2:(w+size)/2]

    return roi


def process_command_line():

    description = "Finds the optimal common registration among a set of images"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("reference", type=str,
            help="Reference image against which the input images will be aligned")
    parser.add_argument("files", nargs='*',
            help="Input image files to be registered")
    parser.add_argument('-r', "--reduction", type=int, default=100,
            help="Reduction percentage with which to register images")
    parser.add_argument('-d', "--work-dir", default='/dev/shm/',
            help="Where to write generated files to")

    return parser.parse_args()


if __name__ == "__main__":
    main()
