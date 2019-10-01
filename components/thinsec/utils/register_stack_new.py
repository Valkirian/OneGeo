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

    # Feature detector
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=opt.contrast_threshold,
                                       sigma=opt.sigma)

    initial_w = opt.strip_width
    s_width = initial_w
    max_width = 3700

    # queryImage
    img1_subg = cvt.color_to_gray(cvt.file_to_cv2(opt.reference))
    target_size = img1_subg.shape[::-1]
    img1_bb = get_bounding_box(img1_subg.shape)
    basename_ref = os.path.basename(opt.reference)
    img1_bands = get_image_bands(img1_subg, max_width)
    debug_log("Gather features of reference image", basename_ref,
                "with w =", max_width)
    kp1, des1 = get_features_on_bands(img1_subg, img1_bands, sift)
    ref_package = (kp1, des1)

    # objects cached by strip width, with initial element
    #img1_features = {}
    match_stats = {'fail': {s_width: 0}, 'success': {s_width: 0}}

    all_bboxes = [img1_bb.tolist(),]
    pre_aligned_files = [opt.reference,]

    # trainImages
    targets = sorted(set(opt.files) - set([opt.reference,]))
    for img_file in targets:

        basename = os.path.basename(img_file)
        img2 = cvt.file_to_cv2(img_file)
        img2_subg = cvt.color_to_gray(img2)
        keypoints = []
        descriptors = []

        s_width = initial_w
        s_clearance = 1000

        converged = False
        while not converged:

            debug_log("Gather features of image", basename, "with w =", s_width,
                      "and c =", s_clearance)
            img2_bands = get_image_bands(img2_subg, s_width)
            kp2, des2 = get_features_on_bands(img2_subg, img2_bands, sift)
            keypoints.extend(kp2)
            descriptors.append(des2)
            target_package = (keypoints, np.vstack(descriptors))

            warp_matrix = match_get_transf(ref_package, target_package,
                                           opt.min_matches)
            print("Warp matrix {}->{}:".format(basename_ref, basename))
            print(warp_matrix)
            success = (warp_matrix is not None and
                       is_good_matrix(warp_matrix, 3e-3))
            converged = success

            if not success:
                debug_log("Not good enough matching achieved.",
                          "Increasing bands width")
                iter_step = 500
                s_width = iter_step
                s_clearance += s_width
                continue

            mark_successful_iter(s_width, match_stats)
            img2_aligned = align_image(img2, warp_matrix, target_size)
            img2_adj_bb = get_adjusted_bounding_box(img1_bb,
                                                    get_bounding_box(img2.shape),
                                                    warp_matrix)
            aligned_file = os.path.join(opt.work_dir, "pre-" + basename)
            cvt.cv2_to_file(img2_aligned, aligned_file)
            all_bboxes.append(img2_adj_bb)
            pre_aligned_files.append(aligned_file)

        result = "done" if success else "failed"
        debug_log("Alignment of", img_file, result)

    common_box = intersect_bounding_boxes(target_size, all_bboxes)

    for fi_aligned in pre_aligned_files:
        debug_log("Cropping", fi_aligned, newline=False)
        aligned = cvt.file_to_cv2(fi_aligned)
        cropped = crop_to_bounding_box(aligned, common_box)

        cf_name = (("reg-" + basename_ref) if fi_aligned == opt.reference else
                   os.path.basename(fi_aligned).replace("pre-", "reg-"))
        cropped_file = os.path.join(opt.work_dir, cf_name)
        success = cvt.cv2_to_file(cropped, cropped_file)

        if success and not opt.keep_uncropped and fi_aligned != opt.reference:
            os.remove(fi_aligned)

        result = "done" if success else "failed"
        print(result)


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


def get_features_on_bands(source_image_gray, bands, detector, name=None):

    keypoints = []
    descriptors = []

    for band in bands:
        image_band = source_image_gray[band[0][0]:band[0][1],
                                       band[1][0]:band[1][1]]
        kp, des = detector.detectAndCompute(image_band, None)

        if des is None:
            return None, None

        descriptors.append(des)
        for keyp in kp:
            new_kp = cv2.KeyPoint(x=keyp.pt[0] + band[1][0],
                                  y=keyp.pt[1] + band[0][0],
                                  _size=keyp.size, _angle=keyp.angle,
                                  _response=keyp.response, _octave=keyp.octave,
                                  _class_id=keyp.class_id)
            keypoints.append(new_kp)

    return keypoints, np.vstack(descriptors)


def match_get_transf(ref_package, target_package, min_matches=10,
                     distance_thd=1400, descriptor_distance_thd=0.7):

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 16)
    search_params = dict(checks = 50)

    kp1, des1 = ref_package
    kp2, des2 = target_package

    debug_log("FLANN matching of", len(kp1), "against", len(kp2), "keypoints")
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    """
    # store all the good matches as per Lowe's ratio test.
    good = [ m for m, n in matches if m.distance < 0.7*n.distance ]
    debug_log("There are", len(good), "good matches")

    if len(good) < min_matches:
        debug_log("Not enough matches are found ({}/{})".format(len(good),
                                                                min_matches))
        return None

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    """

    coll = []

    for m, n in matches:
        src = np.array(kp1[m.queryIdx].pt)
        dst = np.array(kp2[m.trainIdx].pt)
        desc_n_d = m.distance/n.distance
        dist_pts = np.linalg.norm(src - dst)
        coll.append([desc_n_d, dist_pts, src, dst])

    print("Phy dist: min {}, max {}".format(min(c[1] for c in coll if c[0] < descriptor_distance_thd ),
                                            max(c[1] for c in coll if c[0] < descriptor_distance_thd )))
    filtered = [c for c in coll
                if c[0] < descriptor_distance_thd and c[1] < distance_thd]
    if len(filtered) == 0:
        return None
    print("There are {} good matches".format(len(filtered)))

    src_pts = np.array([[c[2],] for c in filtered ])
    dst_pts = np.array([[c[3],] for c in filtered ])

    warp_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 6.0)
    #warp_matrix = cv2.estimateRigidTransform(src_pts, dst_pts, True)

    return warp_matrix


def is_good_matrix(warp_matrix, tol=1e-3):

    diag_divergence = np.r_[1, 1, 1] - warp_matrix.diagonal()
    divergence = np.abs(diag_divergence).max()
    print("Matrix divergence: {:.4g}".format(divergence))

    return (divergence < tol)


def align_image(image, warp_matrix, target_size):

    aligned = cv2.warpPerspective(image, warp_matrix, tuple(target_size),
                                  flags=(cv2.INTER_LINEAR +
                                         cv2.WARP_INVERSE_MAP))
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


def get_next_iter(value, match_stats, iter_step=500):

    match_stats['fail'][value] += 1
    available = sorted(match_stats['success'].keys(), reverse=True)

    if max(available) <= value:
        value += iter_step
        match_stats['fail'][value] = 0
        match_stats['success'][value] = 0
    else:
        current = available.index(value)
        next_v = available[current-1]
        m = match_stats
        if m['fail'][next_v] == 0:
            value = next_v
        else:
            remaining = available[:current]
            rank = sorted( (float(m['success'][k])/m['fail'][k], k)
                          for k in m['success']
                            if (k in remaining and m['fail'][k] > 0))
            value = rank[-1][1]

    return value


def mark_successful_iter(value, match_stats):

    match_stats['success'][value] += 1


def process_command_line():

    description = "Finds the optimal common registration among a set of images"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("reference", type=str,
            help="Reference image against which the input images will be aligned")
    parser.add_argument("files", nargs='*',
            help="Input image files to be registered")
    parser.add_argument('-m', "--min-matches", type=int, default=20,
            help="Minimum number of matches to consider a successful registration")
    parser.add_argument('-w', "--strip-width", type=int, default=700,
            help="image bands width to use for feature extraction")
    parser.add_argument('-e', "--clearance", type=int, default=1000,
            help=("Initical image bands clearance off borders to use "
                  "for feature extraction"))
    parser.add_argument('-s', "--sigma", type=float, default=1.6,
            help="Gaussian blur parameter for SIFT")
    parser.add_argument('-c', "--contrast-threshold", type=float, default=0.1,
            help="Contrast threshold parameter for SIFT")
    parser.add_argument('-d', "--work-dir", default='/dev/shm/',
            help="Where to write generated files to")
    parser.add_argument('-k', "--keep-uncropped", action="store_true",
            help="Avoids erasing of aligned uncropped output images")

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
