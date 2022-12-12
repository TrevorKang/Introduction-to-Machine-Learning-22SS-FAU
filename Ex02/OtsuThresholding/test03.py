import numpy as np

def create_greyscale_histogram(img):
    '''
    returns a histogram of the given image
    :param img: 2D image in greyscale [0, 255]
    :return: np.ndarray (256,) with absolute counts for each possible pixel value
    '''
    # TODO
    hist = np.bincount(img.ravel(), minlength=256)
    return hist

def p_helper(hist, theta: int):
    '''
    Compute p0 and p1 using the histogram and the current theta,
    do not take care of border cases in here
    :param hist:
    :param theta: current theta
    :return: p0, p1
    '''
    # TODO
    total_pixel = sum(hist)
    p0 = 0
    for i in range(0, theta + 1):
        p0 += (hist[i] / total_pixel)
    p1 = 1 - p0
    return p0, p1

