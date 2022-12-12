'''

Created on 05.10.2016
Modified on 09.07.2022
@author: Xingjian KANG[ev00ykob], Jin HUANG[an46ykim] & Lu TIAN[ga94cice]

'''

import numpy as np
import cv2
import matplotlib.pyplot as plt

# do not import more modules!


def drawCircle(img, x, y):
    '''
    Draw a circle at circle of radius 5px at (x, y) stroke 2 px
    This helps you to visually check your methods.
    :param img: a 2d nd-array
    :param y:
    :param x:
    :return: img with circle at desired position
    '''
    cv2.circle(img, (x, y), 5, 255, 2)
    return img


def binarizeAndSmooth(img) -> np.ndarray:
    '''
    First Binarize using threshold of 115, then smooth with gauss kernel (5, 5)
    :param img: grayscale image in range [0, 255]
    :return: preprocessed image
    '''
    ret, img_binary = cv2.threshold(img, 115, 255, cv2.THRESH_BINARY) # ret is a boolean
    ksize = (5, 5)
    sigma = 0
    return cv2.GaussianBlur(img_binary, ksize, sigma)


def drawLargestContour(img) -> np.ndarray:
    '''
    find the largest contour and return a new image showing this contour drawn with cv2 (stroke 2)
    :param img: preprocessed image (mostly b&w)
    :return: contour image
    '''
    # Black background, bright contour
    blurredImage = img.copy()
    contours, hierarchy = cv2.findContours(blurredImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = []
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    max_idx = area.index(max(area))
    largestContour = cv2.drawContours(np.zeros(img.shape), contours, max_idx, (255, 255, 255), 2)
    return largestContour


def getFingerContourIntersections(contour_img, x) -> np.ndarray:
    '''
    Run along a column at position x, and return the 6 intersecting y-values with the finger contours.
    (For help check Palmprint_Algnment_Helper.pdf section 2b)
    :param contour_img:
    :param x: position of the image column to run along
    :return: y-values in np.ndarray in shape (6,)
    '''
    # to keep the indices of all white points
    a = []
    # to keep six contour points we want
    b = []
    for i in range(len(contour_img[:, x])):
        if contour_img[i][x] == 255:
            a.append(i)
    for j in range(2, len(a) - 5):
        # honestly, I don't know how exactly minus 5 comes,
        # maybe it has something to do with the stroke size
        # one thing is clear: smaller than 5, cannot get rid of the border
        if a[j - 1] + a[j + 1] != 2 * a[j] and a[j - 1] != (a[j] - 1):
            if len(b) != 6:
                b.append(a[j])
    return np.array(b)


def findKPoints(img, y1, x1, y2, x2) -> tuple:
    '''
    given two points and the contour image, find the intersection point k
    :param img: binarized contour image (255 == contour)
    :param y1: y-coordinate of point
    :param x1: x-coordinate of point
    :param y2: y-coordinate of point
    :param x2: x-coordinate of point
    :return: intersection point k as a tuple (ky, kx)
    '''
    k = (x2 - x1) / (y2 - y1)
    b = y2 * k - x2
    for x in range(x2 + 1, len(img[0, :])):
        y = int((x + b) / k)
        if img[y, x] == 255:
            break
    kx, ky = x, y
    return ky, kx


def getCoordinateTransform(k1, k2, k3) -> np.ndarray:
    '''
    Get a transform matrix to map points from old to new coordinate system defined by k1-3
    Hint: Use cv2 for this.
    :param k1: point in (y, x) order
    :param k2: point in (y, x) order
    :param k3: point in (y, x) order
    :return: 2x3 matrix rotation around origin by angle
    '''
    y1, x1 = k1
    y2, x2 = k2
    y3, x3 = k3
    # looking for intersection point (90'):
    k13 = (y3 - y1) / (x3 - x1)
    b13 = (x3 * y1 - x1 * y3) / (x3 - x1)  # line between k1 and k3
    k12 = (x1 - x3) / (y3 - y1)
    b12 = y2 - k12 * x2  # line between O and k2
    xo = (b12 - b13) / (k13 - k12)
    yo = k12 * xo + b12  # new center position (yo, xo)
    theta = np.rad2deg(np.arctan((y2 - yo) / (x2 - xo)))  # get the angle of rotation

    return cv2.getRotationMatrix2D(center=(yo, xo), angle=theta, scale=1)


def palmPrintAlignment(img):
    '''
    Transform a given image like in the paper using the helper functions above when possible
    :param img: greyscale image
    :return: transformed image
    '''

    # TODO threshold and blur
    blur = binarizeAndSmooth(img)

    # TODO find and draw largest contour in image
    contour = drawLargestContour(blur)

    # TODO choose two suitable columns and find 6 intersections with the finger's contour
    x1 = 16
    x2 = 20
    intersection_x1 = getFingerContourIntersections(contour_img=contour, x=x1)
    intersection_x2 = getFingerContourIntersections(contour_img=contour, x=x2)

    # TODO compute middle points from these contour intersections
    y11, x11 = (intersection_x1[0] + intersection_x1[1]) / 2, x1
    y12, x12 = (intersection_x2[0] + intersection_x2[1]) / 2, x2

    y21, x21 = (intersection_x1[2] + intersection_x1[3]) / 2, x1
    y22, x22 = (intersection_x2[2] + intersection_x2[3]) / 2, x2

    y31, x31 = (intersection_x1[4] + intersection_x1[5]) / 2, x1
    y32, x32 = (intersection_x2[4] + intersection_x2[5]) / 2, x2

    # TODO extrapolate line to find k1-3
    k1 = findKPoints(img=contour, y1=y11, x1=x11, y2=y12, x2=x12)
    k2 = findKPoints(img=contour, y1=y21, x1=x21, y2=y22, x2=x22)
    k3 = findKPoints(img=contour, y1=y31, x1=x31, y2=y32, x2=x32)

    # TODO calculate Rotation matrix from coordinate system spanned by k1-3
    rotation_matrix = getCoordinateTransform(k1, k2, k3)

    # TODO rotate the image around new origin
    height, width = img.shape
    fixed_image = cv2.warpAffine(img, M=rotation_matrix, dsize=(width, height))
    return fixed_image


if __name__ == '__main__':

    plt.subplots(2, 3)
    arr = ['Hand1.jpg', 'Hand2.jpg', 'Hand3.jpg']
    for i in range(1, 4):
        img = cv2.imread(arr[i - 1], cv2.IMREAD_GRAYSCALE)
        img_aligned = palmPrintAlignment(img)
        plt.subplot(2, 3, i)
        plt.imshow(img, 'gray')
        plt.subplot(2, 3, i + 3)
        plt.imshow(img_aligned, 'gray')
    plt.show()
