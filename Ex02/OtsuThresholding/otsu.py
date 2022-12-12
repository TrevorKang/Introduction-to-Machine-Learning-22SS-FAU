"""
Created on 28.05.2022
@Author: Xingjian Kang

"""

import numpy as np

#
# NO OTHER IMPORTS ALLOWED
#


def create_greyscale_histogram(img):
    '''
    returns a histogram of the given image
    :param img: 2D image in greyscale [0, 255]
    :return: np.ndarray (256,) with absolute counts for each possible pixel value
    '''
    # TODO
    hist = np.bincount(img.ravel(), minlength=256)
    return hist


def binarize_threshold(img, t):
    '''
    binarize an image with a given threshold
    :param img: 2D image as ndarray
    :param t: int threshold value
    :return: np.ndarray binarized image with values in {0, 255}
    '''
    # TODO
    # white 255
    # black 0
    b_image = np.where(img <= t, 0, 255)
    return b_image


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


def mu_helper(hist, theta, p0, p1):
    '''
    Compute mu0 and m1
    :param hist: histogram
    :param theta: current theta
    :param p0:
    :param p1:
    :return: mu0, mu1
    '''
    # TODO
# calculate mu0 with a temporary variable
    temp0 = 0
    for i in range(0, theta + 1):
        temp0 += i * hist[i]
    if p0 == 0:
        mu0 = 0
    else:
        mu0 = temp0 * 1 / p0
# calculate mu1 with a temporary variable
    temp1 = 0
    for i in range(theta + 1, len(hist)):
        temp1 += i * hist[i]
    if p1 == 0:
        mu1 = 0
    else:
        mu1 = temp1 * 1 / p1
    return mu0, mu1


def calculate_otsu_threshold(hist):
    '''
    calculates theta according to otsus method
    :param hist: 1D array
    :return: threshold (int)
    '''
    # TODO
    variance_within = np.zeros([256])
    for i in range(256):
        p0, p1 = p_helper(hist, i)
        mu0, mu1 = mu_helper(hist, i, p0, p1)
        variance_within[i] = p0 * p1 * (mu0 - mu1) ** 2
    threshold = np.argmax(variance_within)
    return threshold


def otsu(img):
    '''
    calculates a binarized image using the otsu method.
    Hint: reuse the other methods
    :param image: grayscale image values in range [0, 255]
    :return: np.ndarray binarized image with values {0, 255}
    '''
    # TODO
    hist = create_greyscale_histogram(img)
    threshold = calculate_otsu_threshold(hist)
    return binarize_threshold(img, threshold)


# if __name__ == '__main__':
#     img = cv2.imread('contrast.jpg', cv2.IMREAD_GRAYSCALE)
#     hist = create_greyscale_histogram(img)
#
#     test_distribution = np.hstack((np.arange(5, 0, -1), np.zeros(246), np.arange(5)))
#     test_distribution2 = np.hstack((np.zeros(123), np.arange(5, 0, -1), np.zeros(123), np.arange(5)))
#
#     thres_1 = calculate_otsu_threshold(test_distribution)
#     thres_2 = calculate_otsu_threshold(test_distribution2)
#     print(thres_1)
#     print(thres_2)
#     'I got the thres_2 as 255, while it should be 127? And fot test_distribution_1 I got 255 too. Why?'
#
#     t = calculate_otsu_threshold(hist)
#     print(t)
#     'This one is correct, I got 120'
# #     t = calculate_otsu_threshold(hist)
# #     print(t)
# #     # theta = 23
# #     # p0, p1 = p_helper(hist,theta)
# #     # m0, m1 = mu_helper(hist, theta, p0, p1)
# #     # print(m0,m1)
