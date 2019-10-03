import numpy as np

def get_inscribed_rectangle(radius, c_x, c_y):

    side = int(round(np.sqrt(2)*radius))
    rectangle = np.r_[c_x, c_y, c_x, c_y] + np.r_[-1, -1, 1, 1]*side/2
    return list(rectangle)

def get_crop(image_shape, rectangle):

    height, width = image_shape

    crop = [float(rectangle[0])/width, float(rectangle[1])/height,
            float(rectangle[2])/width, float(rectangle[3])/height]
    return crop
