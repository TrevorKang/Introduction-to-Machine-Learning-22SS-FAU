import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("guagua.jpg", 0)
kernel = np.ones((5, 5), np.uint8)
img_dilate = cv2.dilate(img, kernel, iterations=1)
img_erosion = cv2.erode(img, kernel, iterations=1)

th2 = cv2.adaptiveThreshold(img, 155, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
plt.imshow(th2, 'gray')
plt.show()

ret2, th3 = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
plt.imshow(th3, 'gray')
plt.show()

plt.subplots(1, 3)
plt.subplot(1, 3, 1)
plt.imshow(img_dilate, 'gray')

plt.subplot(1, 3, 2)
plt.imshow(img_erosion, 'gray')

plt.subplot(1, 3, 3)
plt.imshow(img_dilate - img_erosion, 'gray')
plt.show()

canny = cv2.Canny(img, 50, 155)
plt.imshow(canny,'gray')
plt.show()

# gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
# plt.imshow(gradient, 'gray')
# plt.show()

