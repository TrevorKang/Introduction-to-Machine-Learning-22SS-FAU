"""
Created on 30.05.2022
@Author: HUANG, TIAN

"""

# Implement the histogram equalization in this file
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('hello.png', 0)


def createHistogram(img):
    histogram = np.zeros([256], np.int32)
    for row in range(img.shape[0]):
        for column in range(img.shape[1]):
            histogram[img[row][column]] += 1
    return histogram

#
# plt.plot(createHistogram(img))
# plt.show()

hist = createHistogram(img)
print(sum(hist[:90]))


def pdf(pixel_value: int):
    size = img.shape[0] * img.shape[1]
    px = hist[pixel_value] / size
    return px


def cdf(pixel_value: int):
    cx = 0
    for u in range(pixel_value + 1):
        px = pdf(u)
        cx += px
    return cx


"""Pixels are spread evenly across the entire range of intensity values"""

c = np.zeros(256)
for i in range(256):
    c[i] = cdf(i)
print(sum(c[:90]))

plt.plot(c)
plt.show()

c_min = c[np.argmin(c)]

new_image = np.zeros(img.shape)
for row in range(img.shape[0]):
    for column in range(img.shape[1]):
        old = cdf(img[row][column])
        new_image[row][column] = (old - c_min) * 255 / (1 - c_min)

# pred = np.array(new_image, np.uint8)
# cv2.imshow("kitty.png", pred)
# cv2.waitKey()
cv2.imwrite('kitty.png', new_image)
plt.subplot(1, 2, 1)
plt.imshow(img, 'gray')
plt.title('Original')
plt.subplot(1, 2, 2)
plt.imshow(new_image, 'gray')
plt.title('Fixed')
plt.show()
