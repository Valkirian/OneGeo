import cv2
import numpy as np


def find_left_border(image_grayscale, max_col=500, robust_pick=True):

    strip_left = denoise_strip(image_grayscale[:, :max_col])
    starting_zeros = strip_left[strip_left[:, 0] == 0, :]
    accum = starting_zeros.cumsum(axis=1)
    before = accum[:, :-1]
    after = accum[:, 1:]
    mismatch = (before != after)
    borders = mismatch.argmax(axis=1)

    leftmost_border = return_border(borders, "Left", robust_pick)

    return leftmost_border


def find_top_border(image_grayscale, max_row=500, robust_pick=True):

    strip_top = denoise_strip(image_grayscale[:max_row, :])
    starting_zeros = strip_top[:, strip_top[0, :] == 0]
    accum = starting_zeros.cumsum(axis=0)
    before = accum[:-1, :]
    after = accum[1:, :]
    mismatch = (before != after)
    borders = mismatch.argmax(axis=0)
    topmost_border = return_border(borders, "Top", robust_pick)

    return topmost_border


def find_right_border(image_grayscale, max_col=500, robust_pick=True):

    strip_right = denoise_strip(image_grayscale[:, -max_col:])
    starting_zeros = strip_right[strip_right[:, -1] == 0, :][:, ::-1]
    accum = starting_zeros.cumsum(axis=1)
    before = accum[:, :-1]
    after = accum[:, 1:]
    mismatch = (before != after)
    borders = mismatch.argmax(axis=1)

    border = return_border(borders, "Right", robust_pick)
    leftmost_border = image_grayscale.shape[1] - border

    return leftmost_border


def find_bottom_border(image_grayscale, max_row=500, robust_pick=True):

    strip_bottom = denoise_strip(image_grayscale[-max_row:, :])
    starting_zeros = strip_bottom[:, strip_bottom[-1, :] == 0][::-1, :]
    accum = starting_zeros.cumsum(axis=0)
    before = accum[:-1, :]
    after = accum[1:, :]
    mismatch = (before != after)
    borders = mismatch.argmax(axis=0)

    border = return_border(borders, "Bottom", robust_pick)
    bottommost_border = image_grayscale.shape[0] - border

    return bottommost_border


def get_black_borders(image_grayscale, fraction=0.1, robust_pick=True):

    h, w = image_grayscale.shape[:2]
    left = find_left_border(image_grayscale, int(fraction*w), robust_pick)
    top = find_top_border(image_grayscale, int(fraction*h), robust_pick)
    right = find_right_border(image_grayscale, int(fraction*w), robust_pick)
    bottom = find_bottom_border(image_grayscale, int(fraction*h), robust_pick)

    return (left, right, top, bottom)


class BorderDetector:

    def __init__(self, name, is_vertical, is_from_extreme):

        self.name = name
        self.vertical = is_vertical
        self.from_extreme = is_from_extreme

    def find_border(self, image_grayscale, limit=500, robust_pick=True):

        if self.vertical:
            if self.from_extreme:
                strip = denoise_strip(image_grayscale[:, -limit:])
            else:
                strip = denoise_strip(image_grayscale[:, :limit])
        else:
            if self.from_extreme:
                strip = denoise_strip(image_grayscale[-limit:, :])
            else:
                strip = denoise_strip(image_grayscale[:limit, :])

        i_axis = int(self.vertical)
        accum = strip.cumsum(axis=i_axis)

        # Stop already if strip is non-black along an entire transverse line
        h, w = strip.shape[:2]
        strip_width = w if self.vertical else h
        if accum.max() == strip_width:
            return image_grayscale.shape[i_axis] if self.from_extreme else 0

        if self.vertical:
            starting_zeros = strip[strip[:, 0] == 0, :]
            accum = accum[strip[:, 0] == 0, :]
            if self.from_extreme:
                starting_zeros = starting_zeros[:, ::-1]
            before = accum[:, :-1]
            after = accum[:, 1:]
        else:
            starting_zeros = strip[:, strip[0, :] == 0]
            accum = accum[:, strip[0, :] == 0]
            if self.from_extreme:
                starting_zeros = starting_zeros[::-1, :]
            before = accum[:-1, :]
            after = accum[1:, :]

        mismatch = (before != after)
        borders = mismatch.argmax(axis=i_axis)

        border = return_border(borders, self.name, robust_pick)
        most_border = (border if not self.from_extreme else
                       image_grayscale.shape[i_axis] - border)

        return most_border


def get_black_borders_cls(image_grayscale, fraction=0.1,
                          robust_pick=True):

    h, w = image_grayscale.shape[:2]
    left_d = BorderDetector("Left", True, False)
    right_d = BorderDetector("Right", True, True)
    top_d = BorderDetector("Top", False, False)
    bottom_d = BorderDetector("Bottom", False, True)

    left = left_d.find_border(image_grayscale, int(fraction*w), robust_pick)
    top = top_d.find_border(image_grayscale, int(fraction*h), robust_pick)
    right = right_d.find_border(image_grayscale, int(fraction*w), robust_pick)
    bottom = bottom_d.find_border(image_grayscale, int(fraction*h), robust_pick)

    return (left, right, top, bottom)


def return_border(borders, name, robust_pick=True):

    uniques = np.unique(borders)

    if uniques.shape[0] > 1:
        border = (int(round(get_peak(borders))) if robust_pick
                  else (uniques.max() + 1))
    else:
        print(name, "border seems to be all black!")
        border = 0

    return border


def get_peak(values, bins='auto'):

    frequencies, values = np.histogram(values, bins)

    if len(frequencies) > 1:
        # Omit first bin, which is the starting border
        peak = values[1 + frequencies[1:].argmax()]
    else:
        peak = values[frequencies.argmax()]

    return peak


def crop_borders(image, borders):

    left, right, top, bottom = borders
    return image[top:bottom, left:right, :]


def denoise_strip(strip):

    min_dim, max_dim = sorted(strip.shape[:2])

    close_element = np.ones(tuple(2*[int(0.08*min_dim)]), np.uint8)
    open_element = np.ones(tuple(2*[min(int(0.12*min_dim), max_dim)]), np.uint8)

    no_holes = cv2.morphologyEx(strip, cv2.MORPH_CLOSE, close_element)
    no_noise = cv2.morphologyEx(no_holes, cv2.MORPH_OPEN, open_element)

    return no_noise
