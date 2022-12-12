import numpy as np
import cv2
import matplotlib.pyplot as plt

def binarize_threshold(img, t):
    '''
    binarize an image with a given threshold
    :param img: 2D image as ndarray
    :param t: int threshold value
    :return: np.ndarray binarized image with values in {0, 255}
    '''
    # black, white = 0, 255
    # white_array = np.where((img <= t), img, white)
    # final_array = np.where((white_array > t), white_array, black)
    #
    # return final_array
    b_image = np.where(img <= t, 0, 255)
    return b_image

if __name__ == '__main__':
    img = cv2.imread('contrast.jpg', cv2.IMREAD_GRAYSCALE)
    res = binarize_threshold(img,120)
    if res is not None:
        plt.subplot(1, 2, 2)
        plt.imshow(res, 'gray')
        plt.title('Otsu\'s - Threshold = 120')
    plt.show()