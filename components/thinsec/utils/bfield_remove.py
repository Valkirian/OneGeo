import sys

import cv2
import numpy as np


spaces = ('hsv', 'hls', 'lab')
spaces_transform = {'hls': cv2.COLOR_BGR2HLS,
                    'hsv': cv2.COLOR_BGR2HSV,
                    'lab': cv2.COLOR_BGR2LAB}
spaces_invtransf = {'hls': cv2.COLOR_HLS2BGR,
                    'hsv': cv2.COLOR_HSV2BGR,
                    'lab': cv2.COLOR_LAB2BGR}
spaces_names = {'hls': ('Hue', 'Luminosity', 'Saturation'),
                'hsv': ('Hue', 'Saturation', 'Value'),
                'lab': ('L', 'A', 'B')}
spaces_correct_channel = {'hls': 1,
                          'hsv': 2,
                          'lab': 0}


def main():

    space = sys.argv[1]
    path_bfield = sys.argv[2]
    path_image = sys.argv[3]

    arr_bfield = cv2.imread(path_bfield, cv2.IMREAD_COLOR)
    arr_image = cv2.imread(path_image, cv2.IMREAD_COLOR)

    """
    space_transf = spaces_transform[space]
    h_bfield = cv2.cvtColor(arr_bfield, space_transf)
    h_image = cv2.cvtColor(arr_image, space_transf)

    channels = spaces_names[space]
    show_channels(h_bfield, channels, 'bfield')
    print('\n')
    show_channels(h_image, channels, 'image')

    correction_channel = spaces_correct_channel[space]
    c_bfield = np.ones_like(h_bfield)
    c_bfield[:,:,correction_channel] = h_bfield[:,:,correction_channel]
    c_bfield_n = c_bfield.astype(float)/255
    h_image_n = h_image.astype(float)/255
    #c_image = h_image - c_bfield
    c_image_n = h_image_n/c_bfield_n
    c_image = (255*c_image_n).astype(np.uint8)
    print("c_image_n stats: min {}, max {}".format(c_image.min(), c_image.max()))

    c_img = cv2.cvtColor(c_image, spaces_invtransf[space])
    """

    c_img = arr_image.astype(int) - arr_bfield
    c_img[c_img < 0] = 0
    cv2.imwrite("/dev/shm/corrected.png", c_img.astype('uint8'))

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


def show_channels(image, names, title):

    for name, channel in zip(names, cv2.split(image)):

        """
        window_name = "{}: Channel {}".format(title, name)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, channel)
        """
        file_name = "/dev/shm/{}-ch_{}.png".format(title, name)
        cv2.imwrite(file_name, channel)

        print("Stats for {}: {}".format(file_name, get_stats(channel)))


def get_stats(img):

    stats_out = [img.max(), img.min(), round(100*img.mean())]
    stats_name = ['max', 'min', 'mean']
    stats = dict(zip(stats_name, map(int, stats_out)))

    return stats


if __name__ == "__main__":
    main()
