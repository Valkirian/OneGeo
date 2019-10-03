#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from black_borders_detect import find_left_border, find_top_border
import cv_tools as cvt
import dft_stitch as dsl
from fiji_driver import straighten_crop
import laplace2d_dirich as lp2d


# patch_body_corner_fromfiles('/dev/shm/Fused.png', '/dev/shm/Fused-corner.png')
def patch_body_corner_fromfiles(body_file, corner_file):

    body_mat, body = straighten_crop(body_file, True)
    corner_mat, corner_rot = straighten_crop(corner_file, True)

    return patch_body_corner_inmem(body, body_mat, corner_rot, corner_mat)


def patch_body_corner_inmem(body, body_mat, corner_rot, corner_mat):

    print "Body dim:", body.shape
    print "Corner dim:", corner_rot.shape

    corner_shape = corner_rot.shape[:2]
    c_h, c_w = corner_shape
    body_mat[2, :2] = 0
    body_mat[:2, 2] = 0
    corner_mat[:2, 2] = 0
    corner_mat[2, :2] = 0

    corner_body_rotation = np.dot(body_mat, np.linalg.inv(corner_mat))
    corner = cv2.warpPerspective(corner_rot, corner_body_rotation,
                                 tuple(corner_rot.shape[:2][::-1]))

    g_body_chunk = cvt.color_to_gray(body[:c_h, :c_w, :])
    bl_left = find_left_border(g_body_chunk, c_w)
    bl_top = find_top_border(g_body_chunk, c_h)
    blacks = np.r_[bl_top, bl_left]
    chunk_dim = np.minimum(2*blacks, corner_shape)

    g_body_chunk = g_body_chunk[:chunk_dim[0], :chunk_dim[1]]
    g_corner_chunk = cvt.color_to_gray(corner[:chunk_dim[0], :chunk_dim[1], :])

    corr_peak, peak_val = dsl.get_phasecorr_peak(g_body_chunk, g_corner_chunk)
    mat = dsl.get_translation_matrix(corr_peak)[:2]
    corner_trans = dsl.warp_image(corner[:chunk_dim[0], :chunk_dim[1], :],
                                  mat, tuple(g_body_chunk.shape[::-1]))

    blend_area = np.minimum((1.5*blacks).astype(int), corner_shape)

    body_blend_area = body[:blend_area[0], :blend_area[1], :]
    corner_blend_area = corner_trans[:blend_area[0], :blend_area[1], :]

    weight_gen = lp2d.CornerDirichlet(blend_area[1], blend_area[0],
                                      bl_left, bl_top)
    weight_gen.set_boundaries([[1, 1, 0], [1, 1, 0]])
    solu, residuals = weight_gen.solve()
    corner_weight = weight_gen.get_solution()
    body_weight = 1 - corner_weight
    blend = np.uint8(corner_blend_area*corner_weight[:, :, np.newaxis] +
                     body_blend_area*body_weight[:, :, np.newaxis])
    body[:blend_area[0], :blend_area[1], :] = blend

    return body
