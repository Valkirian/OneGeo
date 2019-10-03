
import cv2
import numpy as np
import os

src_folder = '/dev/shm/temp/'
im_file = src_folder + 'im.jpg'

im = cv2.imread(im_file)

h, s, v = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
# Y, Cr, Cb = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb))

# cv2.imwrite(src_folder + 'Y.jpg', Y)
# cv2.imwrite(src_folder + 'Cr.jpg', Cr)
# cv2.imwrite(src_folder + 'Cb.jpg', Cb)

lut_points = np.asarray([[0, 0], [46, 16], [127,117], [208, 218], [255, 255]])
poly = np.polyfit(lut_points[:, 0], lut_points[:, 1], 5)
lut = np.polyval(poly, np.arange(256))
lut[lut < 0] = 0
lut[lut > 255] = 255
lut = lut.astype(np.uint8)
# plot(lut); plot(np.arange(256)); show()

v_corr = cv2.LUT(v, lut)
# Y_corr = cv2.LUT(Y, lut)

im_corr = cv2.cvtColor(cv2.merge((h, s, v_corr)), cv2.COLOR_HSV2BGR)
# im_corr = cv2.cvtColor(cv2.merge((Y_corr, Cr, Cb)), cv2.COLOR_YCrCb2BGR)
cv2.imwrite(src_folder + 'im_corr.jpg', im_corr)
