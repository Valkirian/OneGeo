#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import sys

import cv2
import matplotlib.pyplot as pl
import numpy as np


def main():

    im0_file = sys.argv[1]
    im1_file = sys.argv[2]
    as_grey = True

    im0_cropped, im1_cropped, M_p, translate_M, reference_pts, target_pts = perform_work(im0_file, im1_file, as_grey, "/dev/shm/test")

    print "perspective:", M_p


def perform_work(im0_file, im1_file, as_grey, output_dir):

    im0 = ImagePointPicker(im0_file)
    im1 = ImagePointPicker(im1_file)

    reference_im = im0.data_orig
    reference_pts = np.float(im0.get_points(as_grey))
    target_pts = np.float(im1.get_points(as_grey))

    """
    #M_h, mask = cv2.findHomography(target_pts, reference_pts, cv2.RANSAC, 5.0)
    #matchesMask = mask.ravel().tolist()
    print "homography:", M_h
    """
    M_p = cv2.getPerspectiveTransform(target_pts, reference_pts)

    matches = range(4)
    """
    draw_matches(im0.data, reference_pts, im1.data, target_pts, matches,
                 output_dir, radius=30, thickness=5)
    """

    translate_M, warped_bbox = adjust_warp(im1.data_orig, M_p)
    M_compound = np.dot(translate_M, M_p)
    w, h = warped_bbox[:,0].max(), warped_bbox[:,1].max()
    corrected_im = cv2.warpPerspective(im1.data_orig, M_compound, (w, h))
    """
    corrected_file = os.path.join(output_dir,
                                  "corrected-uncropped-" + os.path.basename(im1_file))
    cv2.imwrite(corrected_file, corrected_im)
    """

    roi_x, roi_y = get_warped_roi(warped_bbox)
    roi_im = corrected_im[roi_y[0]:roi_y[1], roi_x[0]:roi_x[1]]
    """
    cropped_file = os.path.join(output_dir,
                                "corrected-cropped-" + os.path.basename(im1_file))
    cv2.imwrite(cropped_file, roi_im)
    """

    roi_offset = [roi_x[0], roi_y[0]]
    warped_target_pts = transform_points(target_pts, M_compound) - roi_offset
    roi_r, roi_c = get_matched_rois(reference_im, reference_pts,
                                    roi_im, warped_target_pts, matches,
                                    roi_offset)
    crop_ref_im = reference_im[roi_r[0]:roi_r[1], roi_r[2]:roi_r[3]].copy()
    crop_roi_im = roi_im[roi_c[0]:roi_c[1], roi_c[2]:roi_c[3]].copy()

    matched_file = os.path.join(output_dir,
                                "matched-" + os.path.basename(im0_file))
    cv2.imwrite(matched_file, crop_ref_im)

    matched_file = os.path.join(output_dir,
                                "matched-" + os.path.basename(im1_file))
    cv2.imwrite(matched_file, crop_roi_im)

    return (crop_ref_im, roi_r, crop_roi_im, roi_c, M_p, translate_M,
            reference_pts, warped_target_pts)


class ImagePointPicker(object):

    def __init__(self, im_file):

        self.im_file = im_file
        self.data_orig = cv2.imread(self.im_file, cv2.CV_LOAD_IMAGE_COLOR)

    def get_points(self, as_grey=False):

        if as_grey:
            self.data = cv2.cvtColor(self.data_orig, cv2.COLOR_BGR2GRAY)
            pl.matshow(self.data, cmap=pl.cm.gray)
        else:
            self.data = self.data_orig
            pl.imshow(self.data)

        self.fig = pl.gcf()
        #mng = pl.get_current_fig_manager()
        #mng.window.showMaximized()

        x1, x2, y1, y2 = pl.axis()
        pl.axis('off')

        self.fig.canvas.set_window_title(self.im_file)

        ax = self.fig.gca()

        points = pl.ginput(4, timeout=-1)
        x1 = [p[0] for p in points]
        y1 = [p[1] for p in points]

        pl.plot(x1, y1, '.c-')
        ax.figure.canvas.draw()

        return points


def draw_matches(image0, keypoints0, image1, keypoints1, matches, output_dir,
                 **kwargs):

    kp0 = [cv2.KeyPoint(k[0][0], k[0][1], 10) for k in keypoints0]
    kp1 = [cv2.KeyPoint(k[0][0], k[0][1], 10) for k in keypoints1]

    img_matches = drawMatches_shim(image0, kp0, image1, kp1, matches, **kwargs)
    cv2.imwrite(os.path.join(output_dir, "match.jpg"), img_matches)


def drawMatches_shim(img1, kp1, img2, kp2, matches, **kwargs):
    """
    From: http://stackoverflow.com/a/26227854/706031

    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    color = kwargs.pop('color', (255, 255, 0))      # cyan by default
    radius = kwargs.pop('radius', 10)
    thickness = kwargs.pop('thickness', 1)

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    """
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
    """
    for i, mat in enumerate(matches):
        # x - columns
        # y - rows
        (x1,y1) = kp1[i].pt
        (x2,y2) = kp2[mat].pt

        # Draw a small circle at both co-ordinates
        cv2.circle(out, (int(x1),int(y1)), radius, color, thickness)
        cv2.circle(out, (int(x2)+cols1,int(y2)), radius, color, thickness)

        # Draw a line in between the two points
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), color, thickness)

    return out


def transform_points(points, M):

    transformed = []

    for p in points:
        x, y = p
        x_t = (M[0,0]*x + M[0,1]*y + M[0,2])/(M[2,0]*x + M[2,1]*y + M[2,2])
        y_t = (M[1,0]*x + M[1,1]*y + M[1,2])/(M[2,0]*x + M[2,1]*y + M[2,2])

        transformed.append([x_t, y_t])

    return np.array(transformed).round().astype(int)


def get_bounding_box(shape):

    sizes = shape[:2]
    return np.array([[0, 0], [sizes[1], 0], [0, sizes[0]], list(sizes[::-1])])


def adjust_warp(image, M):

    target_bounding_box = transform_points(get_bounding_box(image.shape), M)

    desp_x = -target_bounding_box[:,0].min()
    desp_y = -target_bounding_box[:,1].min()

    translate_M = np.array([[1, 0, desp_x], [0, 1, desp_y], [0, 0, 1]],
                           dtype='float64')
    warped_bbox = transform_points(get_bounding_box(image.shape),
                                   np.dot(translate_M, M))

    return translate_M, warped_bbox


def get_warped_roi(warped_bounding_box):

    _, min_x, max_x, _ = sorted(warped_bounding_box[:,0])
    _, min_y, max_y, _ = sorted(warped_bounding_box[:,1])

    roi_x, roi_y = [min_x, max_x], [min_y, max_y]

    return roi_x, roi_y


def get_distances_to_edges(point, image):

    d_to_left_edge = point[0]
    d_to_top_edge = point[1]
    d_to_right_edge = image.shape[1] - point[0]
    d_to_bottom_edge = image.shape[0] - point[1]

    return np.r_[d_to_left_edge, d_to_top_edge, d_to_right_edge, d_to_bottom_edge]


def get_matched_rois(ref_im, reference_pts, cropped_im, target_pts, matches,
                     roi_offset, match_index=0):

    ref_pts = reference_pts.round().astype(int)

    target_point = target_pts[match_index]
    reference_point = ref_pts[matches[match_index]]

    dist_ref = get_distances_to_edges(reference_point, ref_im)
    dist_crp = get_distances_to_edges(target_point, cropped_im)

    diff_left, diff_top, diff_right, diff_bottom = dist_ref - dist_crp

    if diff_left > 0:
        roi_ref_left = diff_left
        roi_crp_left = 0
    else:
        roi_ref_left = 0
        roi_crp_left = -diff_left

    if diff_top > 0:
        roi_ref_top = diff_top
        roi_crp_top = 0
    else:
        roi_ref_top = 0
        roi_crp_top = -diff_top

    if diff_right > 0:
        roi_ref_right = ref_im.shape[1] - roi_ref_left - diff_right
        roi_crp_right = cropped_im.shape[1]
    else:
        roi_ref_right = ref_im.shape[1]
        roi_crp_right = cropped_im.shape[1] - roi_crp_left + diff_right

    if diff_bottom > 0:
        roi_ref_bottom = ref_im.shape[0] - roi_ref_top - diff_bottom
        roi_crp_bottom = cropped_im.shape[0]
    else:
        roi_ref_bottom = ref_im.shape[0]
        roi_crp_bottom = cropped_im.shape[0] - roi_crp_top + diff_bottom

    roi_ref = [roi_ref_top, roi_ref_bottom, roi_ref_left, roi_ref_right]
    roi_crp = [roi_crp_top, roi_crp_bottom, roi_crp_left, roi_crp_right]

    return roi_ref, roi_crp


if __name__ == "__main__":
    main()

# for i in $(ls *.JPG); do convert -resize 70% -compress jpeg $i output/$i; done
