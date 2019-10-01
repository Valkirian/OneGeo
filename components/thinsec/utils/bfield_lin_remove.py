#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import sys

from cv_tools import(file_to_cv2, cv2_to_file, is_image_dark,
                     blankfield_linear_correct, blankfield_dark_correct)


def main():

    image_file = sys.argv[1]
    bfield_file = sys.argv[2]

    image = file_to_cv2(image_file)
    bfield = file_to_cv2(bfield_file)

    bfield_dark = is_image_dark(bfield)
    print("Is bfield dark: {}".format(bfield_dark))

    corrct = (blankfield_linear_correct(image, bfield) if bfield_dark else
              blankfield_dark_correct(image, bfield))

    img_out = os.path.abspath(os.path.splitext(os.path.basename(image_file))[0]
                              + "-bfield.jpg")
    cv2_to_file(corrct, img_out, 99)
    print("corrected image written to: {}".format(img_out))


if __name__ == "__main__":
    main()
