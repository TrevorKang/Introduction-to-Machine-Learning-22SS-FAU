import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("HandKXJ.jpg", cv2.IMREAD_GRAYSCALE)
ret, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
smooth = cv2.GaussianBlur(thresh, (5, 5), 0)
plt.imshow(smooth, 'gray')
plt.show()

blur = smooth.copy()
whiteImg = np.zeros(img.shape)
contours, hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
area = []
for k in range(len(contours)):
    area.append(cv2.contourArea(contours[k]))
max_idx = area.index(max(area))

img2 = cv2.drawContours(np.zeros(img.shape), contours, max_idx, (255, 255, 255), 2)
plt.imshow(img2, 'gray')
plt.show()
# cv2.imshow("xx", img2)
# cv2.waitKey()

# contour_x = img2[:, 16]
# contour_x = np.pad(img2[:, 16], [1, 1], 'constant', constant_values=255)
# temp = np.where(contour_x == 255) # return the indices of white points
# res = temp[0]
# area = []
# for i in range(len(res) - 1):
#     if res[i] + 1 != res[i + 1]:
#         area.append(res[i]-1)
#         area.append(res[i + 1]-1)
# print(area)
# y = []
# for i in [1, 4, 5, 8, 9, 12]:
#     y.append(area[i])
# y=np.array(y)
# print(y) # [ 41  79 118 137 170 223]

# [  0   1   2  42  43  44  45  78  79  80 119 120 121 136 137 138 171 172
# 173 174 221 222 223 224 239 240 241] result with padding
# [  0   1  41  42  43  44  77  78  79 118 119 120 135 136 137 170 171 172
# 173 220 221 222 223 238 239] result without padding
# x = 16
# a = []
# b = []
# for i in range(len(img2[:, x])):
#     if img2[i][x] == 255:
#         a.append(i)
# print(a)
# for j in range(2, len(a) - 2):
#     if a[j-1] + a[j+1] != 2*a[j] and a[j - 1] != (a[j] - 1):
#         b.append(a[j])
# print(np.array(b))


# img = np.zeros((108, 11))
# img[0:4] = img[13:18] = img[28:33] = img[43:48] = img[58:63] = img[73:78] = img[88:93] = img[103:108] = 255
# img_2 = np.zeros((128, 11))
# img_2[0:4] = 255
# img_2[20:128, 0:11] = img


# img_1 = 255 * np.sum(np.array([np.eye(20, 20, k=i) for i in [-1, 0, 1]]), axis=0)
#
# x = 1
# while True:
#     y = (x - 1) * (12 - 13) / (1 - 4) + 12
#     if img_1[y][x] == 255:
#         break
#     x += 1
# print(y)
# print(x)

# x = 10
# a = []
# b = []
# for i in range(len(img_2[:, x])):
#     if img_2[i][x] == 255:
#         a.append(i)
# for j in range(2, len(a) - 6):
#     if a[j-1] + a[j+1] != 2*a[j] and a[j - 1] != (a[j] - 1):
#         if len(b) != 6:
#             b.append(a[j])
# print(np.array(b))

