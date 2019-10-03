import sys
import os

import cv2
import numpy as np

def main():

    source_path = sys.argv[1]
    source_dir = os.path.dirname(source_path)
    img_orig = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
    circles, cimg, rect_crop = process_image_gray(img_orig)

    #cv2.namedWindow('blurred image, median', cv2.WINDOW_NORMAL)
    #cv2.imshow('blurred image, median', cv2.medianBlur(img, blur_radius))

    cv2.namedWindow('detected circles', cv2.WINDOW_NORMAL)
    cv2.imshow('detected circles', cimg)

    cv2.namedWindow('rectangular crop', cv2.WINDOW_NORMAL)
    cv2.imshow('rectangular crop', rect_crop)
    cropped_path = os.path.join(source_dir, 'rectangular-crop.jpg')
    cv2.imwrite(cropped_path, rect_crop)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_image_gray(img_orig):

    cimg = cv2.cvtColor(img_orig, cv2.COLOR_GRAY2BGR)
    print("Image size: {}".format(img_orig.shape))

    circles = get_single_circle(img_orig, 51, 100, 400)
    for i in circles[0,:]:
        center = (i[0], i[1])
        radius = int(round(0.99*i[2]))
        print("Center: ({c[0]}, {c[1]}), Radius: {c[2]}".format(c=i))

        # Get inscribed rectangle
        rectangle = get_inscribed_rectangle(radius, *center)
        print("Inscribed rect at endpoints {} and {}".format(rectangle[0:2],
                                                             rectangle[2:]))
        rect_crop = img_orig[rectangle[1]:rectangle[3],
                             rectangle[0]:rectangle[2]]

        # Get crop numbers
        crop = get_crop(img_orig.shape, rectangle)
        printed_crop = ', '.join("{:.5f}".format(i) for i in crop)
        print("Crop coordinates: " + printed_crop)

        # Test denormalizing crop
        restored_crop = restore_crop(np.r_[map(float, printed_crop.split(', '))],
                                     img_orig.shape)
        matches = (rectangle == restored_crop).all()
        print("Restored Crop coordinates: {} (matches? {})".format(restored_crop,
                                                                   matches))

        # draw the outer circle
        cv2.circle(cimg, center, radius, (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, center, 2, (0, 0, 255), 3)
        # draw the inscribed rectangle
        cv2.rectangle(cimg, tuple(rectangle[0:2]), tuple(rectangle[2:]),
                      (0, 200, 200), 2)

    return circles, cimg, rect_crop


def get_single_circle(image, blur_radius, canny_threshold, start_param2):

    found = False
    param2 = start_param2

    while not found:
        img = cv2.medianBlur(image, blur_radius)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1,
                                   img.shape[0]/8, param1=canny_threshold,
                                   param2=param2,
                                   minRadius=1000, maxRadius=0)
        found = (circles is not None) and (circles.shape[0] == 1)
        print param2, found
        param2 /= 2

    circles_int = np.uint16(np.around(circles))

    return circles_int

def get_inscribed_rectangle(radius, c_x, c_y):

    side = int(round(np.sqrt(2)*radius))
    rectangle = np.r_[c_x, c_y, c_x, c_y] + np.r_[-1, -1, 1, 1]*side/2
    return rectangle


def get_crop(image_shape, rectangle):

    height, width = image_shape

    denorm = np.r_[width, height, width, height].astype(float)
    crop = rectangle/denorm

    return crop


def restore_crop(crop, image_shape):

    height, width = image_shape

    denorm = np.r_[width, height, width, height].astype(float)
    restored_crop = (crop*denorm).round().astype(int)

    return restored_crop


if __name__ == "__main__":
    main()
