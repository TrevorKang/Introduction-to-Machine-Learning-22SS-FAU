import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import convolve



#
# NO MORE MODULES ALLOWED
#


def gaussFilter(img_in, ksize, sigma):
    """
    filter the image with a gauss kernel
    :param img_in: 2D greyscale image (np.ndarray)
    :param ksize: kernel size (int)
    :param sigma: sigma (float)
    :return: (kernel, filtered) kernel and gaussian filtered image (both np.ndarray)
    """
    # TODO
    kernel = np.zeros((ksize, ksize))
    for i in range(0, ksize):
        for j in range(0, ksize):
            x2 = (i - (ksize - 1) / 2.) ** 2
            y2 = (j - (ksize - 1) / 2.) ** 2
            kernel[i][j] = np.exp(-1 * (x2 + y2) / (2 * (sigma ** 2))) / (2 * np.pi * sigma * sigma)
    kernel /= np.sum(kernel)
    temp = convolve(img_in, kernel)
    convolved = np.asarray(temp, dtype=int)
    return kernel, convolved


def sobel(img_in):
    """
    applies the sobel filters to the input image
    Watch out! scipy.ndimage.convolve flips the kernel...

    :param img_in: input image (np.ndarray)
    :return: gx, gy - sobel filtered images in x- and y-direction (np.ndarray, np.ndarray)
    """
    # TODO
    gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    gy = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])
    gy = np.flipud(gy)  # flip the kernel

    i_x = np.asarray(convolve(img_in, gx), dtype=int)
    i_y = np.asarray(convolve(img_in, gy), dtype=int)

    return i_x, i_y


def gradientAndDirection(gx, gy):
    """
    calculates the gradient magnitude and direction images
    :param gx: sobel filtered image in x direction (np.ndarray)
    :param gy: sobel filtered image in x direction (np.ndarray)
    :return: g, theta (np.ndarray, np.ndarray)
    """
    # TODO
    g = np.asarray(np.sqrt(gx * gx + gy * gy), dtype=int)
    theta = np.arctan2(gy, gx)
    return g, theta


def convertAngle(angle):
    """
    compute nearest matching angle
    :param angle: in radians
    :return: nearest match of {0, 45, 90, 135}
    """
    # TODO
    # convert the angle within [0,180]
    degree = angle * 180 / np.pi
    temp = degree // 180
    if temp != 0:
        degree = degree - 180 * temp
    if 22.5 <= degree < 67.5:
        degree = 45
    elif 67.5 <= degree < 112.5:
        degree = 90
    elif 112.5 <= degree < 157.5:
        degree = 135
    else:
        degree = 0
    return degree


def maxSuppress(g, theta):
    """
    calculate maximum suppression
    :param g:  (np.ndarray)
    :param theta: 2d image (np.ndarray)
    :return: max_sup (np.ndarray)
    """
    # TODO Hint: For 2.3.1 and 2 use the helper method above
    res = np.zeros(np.shape(g))
    for row in range(1, len(theta)-1):  # height
        for column in range(1, len(theta[0])-1):  # width
            degree = convertAngle(theta[row][column])
            if degree == 0:  # horizont
                if g[row][column] == max(g[row][column-1], g[row][column], g[row][column+1]):
                    res[row][column] = g[row][column]
                else:
                    res[row][column] = 0
            elif degree == 45:  # down left to up right
                if g[row][column] == max(g[row-1][column+1], g[row][column], g[row+1][column-1]):
                    res[row][column] = g[row][column]
                else:
                    res[row][column] = 0
            elif degree == 90:  # vertical
                if g[row][column] == max(g[row+1][column], g[row][column], g[row-1][column]):
                    res[row][column] = g[row][column]
                else:
                    res[row][column] = 0
            elif degree == 135:  # down right to up left
                if g[row][column] == max(g[row+1][column+1], g[row][column], g[row-1][column-1]):
                    res[row][column] = g[row][column]
                else:
                    res[row][column] = 0
    return res


def hysteris(max_sup, t_low, t_high):
    """
    calculate hysteris thresholding.
    Attention! This is a simplified version of the lectures hysteresis.
    Please refer to the definition in the instruction

    :param max_sup: 2d image (np.ndarray)
    :param t_low: (int)
    :param t_high: (int)
    :return: hysteris thresholded image (np.ndarray)
    """
    # TODO
    height, width = np.shape(max_sup)
    threshimg = np.zeros((height, width))
    newImage = np.zeros((height, width))

    # classification
    for i in range(height):
        for j in range(width):
            if max_sup[i, j] <= t_low:
                threshimg[i, j] = 0
            elif t_low < max_sup[i, j] <= t_high:
                threshimg[i, j] = 1
            elif max_sup[i, j] > t_high:
                threshimg[i, j] = 2
    # padding, otherwise out of boundary
    padded = np.pad(threshimg, ((1, 1), (1, 1)), 'constant')
    for i in range(1, height + 1):
        for j in range(1, width + 1):
            if padded[i, j] == 2:
                newImage[i - 1, j - 1] = 255
            elif padded[i, j] == 1:
                if padded[i - 1:i + 2, j - 1:j + 2].__contains__(2):
                    newImage[i - 1, j - 1] = 255
    return newImage


def canny(img):
    # gaussian
    kernel, gauss = gaussFilter(img, 5, 2)

    # sobel
    gx, gy = sobel(gauss)

    # plotting
    plt.subplot(2, 2, 1)
    plt.imshow(gx, 'gray')
    plt.title('gx')
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.imshow(gy, 'gray')
    plt.title('gy')
    plt.colorbar()


    # gradient directions
    g, theta = gradientAndDirection(gx, gy)

    # plotting
    plt.subplot(2, 2, 3)
    plt.imshow(g, 'gray')
    plt.title('gradient magnitude')
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.imshow(theta)
    plt.title('theta')
    plt.colorbar()
    plt.show()

    plt.subplot(1, 3, 1)
    plt.title('Unfixed')
    plt.imshow(img, 'gray')
    # maximum suppression
    plt.subplot(1, 3, 2)
    maxS_img = maxSuppress(g, theta)
    plt.title('Maximum Suppression')
    plt.imshow(maxS_img, 'gray')

    # hysteris thresholding
    plt.subplot(1, 3, 3)
    result = hysteris(maxS_img, 50, 115)
    plt.title('Hysteris Thresholding')
    plt.imshow(result, 'gray')
    plt.show()

    return result


if __name__ == '__main__':
    img = cv2.imread('contrast.jpg', 0)
    canny(img)
