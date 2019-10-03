# -*- coding: utf-8 -*-

import cv2
import numpy as np
from skimage.feature import register_translation


def phase_correlation(im0, im1):
    sz0 = np.array(im0.shape)
    sz1 = np.array(im1.shape)
    sz0_2 = 2**np.ceil(np.log2(sz0)).astype(int)
    sz1_2 = 2**np.ceil(np.log2(sz1)).astype(int)
    size = tuple(np.fmax(sz0_2, sz1_2))
    dft0 = np.fft.rfft2(im0, size)
    dft1 = np.fft.rfft2(im1, size)
    prod = dft0 * dft1.conj()
    phase = prod/np.abs(prod)

    return np.fft.irfft2(phase), size

# Perform DFT registration with a default 1/20th pixel sub-sampling
def get_phasecorr_peak(im0, im1, subsampling=20):
    if np.array_equal(im0, im1):
        return (0, 0)
    peak_loc, peak_val, _ = register_translation(im0, im1, subsampling)

    return peak_loc, peak_val


def get_phasecorr_peak__old(im0, im1):
    if np.array_equal(im0, im1):
        return (0, 0)
    h, w = im0.shape
    phasec, size2 = phase_correlation(im0, im1)
    phasec[0, :] = 0
    phasec[:, 0] = 0
    peak_loc = np.array(np.unravel_index(phasec.argmax(), phasec.shape))
    peak_loc[0] = peak_loc[0] if peak_loc[0] < h/2 else peak_loc[0] - size2[0]
    peak_loc[1] = peak_loc[1] if peak_loc[1] < w/2 else peak_loc[1] - size2[1]

    return peak_loc, phasec.max()


def stitching_parts(im0, im1):
    im0_g = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    im1_g = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    corr_peak, peak_val = get_phasecorr_peak(im0_g, im1_g)
    mat = get_translation_matrix(corr_peak)
    trans_M, im1_nbb = adjust_warp(im1, mat)
    im0_nbb = transform_points(get_bounding_box(im0_g.shape), trans_M)
    all_bb = np.vstack([im0_nbb, im1_nbb])
    max_x = all_bb[:,0].max()
    max_y = all_bb[:,1].max()
    size = (max_x, max_y)
    half_0 = cv2.warpAffine(im0, trans_M[:-1].astype(float), size)
    half_1 = cv2.warpAffine(im1, np.dot(trans_M, mat)[:-1].astype(float), size)

    return (max_y, max_x), half_0, half_1, im0_nbb, im1_nbb, mat, trans_M


def get_translation_matrix(corr_peak):
    delta_y, delta_x = corr_peak
    mat = np.array([[1, 0, delta_x], [0, 1, delta_y], [0, 0, 1]], dtype=float)

    return mat


def get_bounding_box(shape, origin=[0, 0]):
    """
    Gets the top-left and bottom-right bounding corners of an image.
    param shape: (height, width) image specification
    param origin: [x0, y0] coordinate of top-left corner of image
    returns: [x, y] coordinates of top-left and bottom-right corners
    """

    sizes = np.array([shape[1], shape[0]])
    zero_ref_bbox = np.array([[0, 0], sizes])
    orig = np.array(origin)
    # Numpy's broadcasting sums origin to the rows of zero_ref_bbox
    return zero_ref_bbox + orig


def warp_image(image, warp_matrix, target_size):
    size_x, size_y = target_size
    aligned = cv2.warpAffine(image, warp_matrix, (size_x, size_y), flags=(cv2.INTER_LINEAR))

    return aligned


def adjust_warp(image, M):
    target_bounding_box = transform_points(get_bounding_box(image.shape), M)
    min_x = target_bounding_box[:,0].min()
    desp_x = 0 if min_x >= 0 else -min_x
    min_y = target_bounding_box[:,1].min()
    desp_y = 0 if min_y >= 0 else -min_y
    translate_M = np.array([[1, 0, desp_x], [0, 1, desp_y], [0, 0, 1]], dtype=int)
    warped_bbox = transform_points(get_bounding_box(image.shape), np.dot(translate_M, M))

    return translate_M, warped_bbox


def transform_points(points, M):
    transformed = []
    is_affine = M.shape[0] == 2
    for p in points:
        x, y = p
        det = 1 if is_affine else float(M[2,0])*x + float(M[2,1])*y + M[2,2]
        x_t = (M[0,0]*x + M[0,1]*y + M[0,2]) / det
        y_t = (M[1,0]*x + M[1,1]*y + M[1,2]) / det
        transformed.append([x_t, y_t])

    return np.array(transformed).round().astype(int)


def linear_blend_weights(im0_bb, im1_bb, weight_shape):
    wgt0 = np.zeros(weight_shape, dtype=bool)
    wgt1 = np.zeros(weight_shape, dtype=bool)
    wgt0[im0_bb[0, 1]:im0_bb[-1, 1], im0_bb[0, 0]:im0_bb[-1, 0], :] = True
    wgt1[im1_bb[0, 1]:im1_bb[-1, 1], im1_bb[0, 0]:im1_bb[-1, 0], :] = True
    intersection = wgt0 * wgt1
    it_y, it_x, _ = np.where(intersection)
    min_x, max_x = it_x.min(), 1+it_x.max()
    min_y, max_y = it_y.min(), 1+it_y.max()
    it_kinks = np.array([[min_x, min_y], [max_x, max_y]])
    is_horizontal = np.diff(it_kinks, axis=0).argmin() == 0
    print ("Stitch is", "horizontal" if is_horizontal else "vertical")
    print (it_kinks)
    xx, yy = np.meshgrid(np.linspace(0, 1, max_x - min_x), np.linspace(0, 1, max_y - min_y))
    transition = (1 - xx) if is_horizontal else (1 - yy)
    blend0 = wgt0.astype(float)
    blend1 = wgt1.astype(float)
    blend0[min_y:max_y, min_x:max_x, :] = transition[:, :, np.newaxis]
    blend1[min_y:max_y, min_x:max_x, :] = 1 - transition[:, :, np.newaxis]

    return blend0, blend1


if __name__ == "__main__":
    import sys
    im0 = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    im1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
    sz, h0, h1, im0_bb, im1_bb, mat, trans_mat = stitching_parts(im0, im1)
    wgt0, wgt1 = linear_blend_weights(im0_bb, im1_bb, h0.shape)
    h_lin = np.uint8(wgt0 * h0 + wgt1 * h1)
    cv2.imwrite(sys.argv[3], h_lin)
