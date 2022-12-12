import numpy as np
import cv2
import matplotlib.pyplot as plt
# k = np.array([1, 2, 3, 4, 5])
# t = np.where(k > 3, 8, 1)
# print(t)

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
    b_image = np.where(img >= t, 255, 0)
    return b_image

img = cv2.imread('contrast.jpg', cv2.IMREAD_GRAYSCALE)
hist = np.bincount(img.ravel(), minlength=256)
t = 120


# total_pixel = sum(hist)
# print(total_pixel)
# p0 = 0
# for i in range(0, 187):
#     p0 += (hist[i] / total_pixel)
# p1 = 1 - p0
# print(p0, p1)
plt.hist(img.ravel(), 256, [0, 256])
plt.show()

# t =np.zeros([256])
# k = np.array([1,3,8,5,8,6])
# print(np.argmax(k))
