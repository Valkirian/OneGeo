#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import multiprocessing
import itertools as it
import sys

import cv2


def main():

    in_file = sys.argv[1]
    num_features = int(sys.argv[2])
    proc_pool = multiprocessing.Pool(6)

    contrastThr = [0.01, 0.04, 0.08, 0.1]
    edgeThr = [2, 10, 20, 50]
    sigma = [0.5, 1.6, 3, 5, 10]

    sweep = list(it.product(contrastThr, edgeThr, sigma))
    proc_pool.map(SiftTester(in_file, num_features), sweep)


class SiftTester(object):

    def __init__(self, in_file, num_features):

        self.in_file = in_file
        self.num_features = num_features

    def __call__(self, vector=(0.04, 10, 1.6)):

        contrastThreshold, edgeThreshold, sigma = vector

        img = cv2.imread(self.in_file)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT(nfeatures=self.num_features,
                        contrastThreshold=contrastThreshold,
                        edgeThreshold=edgeThreshold, sigma=sigma)
        keypoints = sift.detect(img_gray, None)

        param_string = "-sift-c_{}-e_{}-s_{}.jpg".format(contrastThreshold,
                                                         edgeThreshold, sigma)
        out_file = self.in_file.split('.')[0] + param_string

        """
        for k in keypoints:
            k.size = 400
        img_sift = cv2.drawKeypoints(img_gray, keypoints,
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(out_file, img_sift)
        """

        draw_keypoints(img_gray, keypoints, 30, (0, 255, 0), out_file)


def draw_keypoints(image, keypoints, radius, color, out_file):

    img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for k in keypoints:
        cv2.circle(img, tuple(map(int, map(round, k.pt))), radius, color, -1)

    cv2.imwrite(out_file, img)


if __name__ == "__main__":
    main()
