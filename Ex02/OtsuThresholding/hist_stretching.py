import numpy as np
import cv2
import matplotlib.pyplot as plt

"""
    without percentile
"""


def createHistogram(img):
    histogram = np.zeros([256], np.int32)
    for row in range(img.shape[0]):
        for column in range(img.shape[1]):
            histogram[img[row][column]] += 1
    return histogram


def rescale(img, fmin, fmax):
    new_image = np.zeros(img.shape)
    for row in range(img.shape[0]):
        for column in range(img.shape[1]):
            new_image[row][column] = (img[row][column] - fmin) / (fmax - fmin)
    return new_image


def stretching(img, low, high):
    down_limit = img.shape[0] * img.shape[1] * low / 100
    upper_limit = img.shape[0] * img.shape[1] * high / 100

    arr = sorted(img.flatten(), reverse=0)
    fmin = arr[int(down_limit)]
    fmax = arr[int(upper_limit)]
    print(fmin)
    print(fmax)

    new_image = rescale(img, fmin, fmax)
    return new_image


if __name__ == '__main__':
    img = cv2.imread("hello.png", 0)
    fmin = np.min(img.flatten())
    fmax = np.max(img.flatten())
    # print(fmin)
    # print(fmax)

    plt.subplot(1, 3, 1)
    plt.imshow(img, 'gray')
    plt.title('Unfixed')

    fixed_image1 = rescale(img, fmin, fmax)
    plt.subplot(1, 3, 2)
    plt.imshow(fixed_image1, 'gray')
    plt.title('Fixed')

    fixed_image2 = stretching(img, 10, 90)
    plt.subplot(1, 3, 3)
    plt.imshow(fixed_image2, 'gray')
    plt.title('Percentile')

    plt.show()

    # plt.subplot(2, 2, 3)
    # plt.hist(img.flatten(), bins=64)
    # plt.subplot(2, 2, 4)
    # plt.hist(fixed_image.flatten(), bins=64)
    # plt.show()
