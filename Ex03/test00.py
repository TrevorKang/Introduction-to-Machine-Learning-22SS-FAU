import numpy as np
import matplotlib.pyplot as plt
import cv2

# img = cv2.imread('Hand1.jpg', cv2.IMREAD_GRAYSCALE)
# ret, img_binarized = cv2.threshold(img, 115, 255, 0)
# plt.subplot(1, 2, 1)
# plt.imshow(img, 'gray')
# plt.subplot(1, 2, 2)
# # draw contours
# contours, hierarchy = cv2.findContours(img_binarized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# image_contours = cv2.drawContours(img_binarized, contours, 1, (0, 0, 255), 3)
# plt.imshow(image_contours, 'gray')
# plt.show()

# ksize = (5, 5)
# sigma = 2
# blur = cv2.GaussianBlur(image_contours, ksize, sigma)
# plt.imshow(blur, 'gray')
# plt.show()

# x1 = 10
# y1 = 11
#
# x2 = 12
# y2 = 13
#
# x3 = 15
# y3 = 16
#
# if (y3 - y2) / (y1 - y2) == (x3-x2) / (x1 - x2):
#     print("True")

def getFootPoint(point, line_p1, line_p2):
    """
    @point, line_p1, line_p2 : [x, y, z]
    """
    x0 = point[0]
    y0 = point[1]
    z0 = point[2]

    x1 = line_p1[0]
    y1 = line_p1[1]
    z1 = line_p1[2]

    x2 = line_p2[0]
    y2 = line_p2[1]
    z2 = line_p2[2]

    k = -((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1) + (z1 - z0) * (z2 - z1)) / \
        ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)*1.0

    xn = k * (x2 - x1) + x1
    yn = k * (y2 - y1) + y1
    zn = k * (z2 - z1) + z1

    return (xn, yn, zn)


img = np.array([[1,2,3],
                [4,5,6],
                [7,8,9]])
for row in range(len(img)):
    for col in range(len(img[row, :])):
        img[row][col] += 1
print(img)

x1 = 1
x2 = 4
y1 = 12
y2 = 13
k = (y2 - y1) / (x2 - x1)
b = y2 - k * x1
print(k)
print(b)