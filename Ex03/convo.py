from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def make_kernel(ksize, sigma):
    """

    :param ksize: kernel size, k * k
    :param sigma: standard variance
    :return: kernel matrix
    """
    kernel = np.zeros((ksize, ksize))  # set up an array to storage the value
    for i in range(0, ksize):
        for j in range(0, ksize):
            x2 = (i - (ksize - 1) / 2.) ** 2
            y2 = (j - (ksize - 1) / 2.) ** 2
            kernel[i][j] = np.exp(-1 * (x2 + y2) / (2 * (sigma ** 2))) / (2 * np.pi * sigma * sigma)
    return kernel / np.sum(kernel)  # normalization to 1


def slow_convolve(arr, k):
    """
    :param arr: input image as ndarray
    :param k: filter/kernel
    :return: convolved image, with the same size as the origin one

    kernel size: kw * kh
    image size: W * H
    padding: pw * ph, pw = kw // 2
    output image size: (W + 2 * pw - kw + 1) * (H + 2 * ph - kh +1)

    """
    res = np.zeros(np.shape(arr))
    kh, kw = np.shape(k)
    h, w = np.shape(arr)
    ph = kh // 2
    pw = kw // 2
    pad = np.pad(arr, ((ph, ph), (pw, pw)), 'constant')
    for i in range(0, h, 1):
        for j in range(0, w, 1):
            input = pad[i: i + kh, j: j + kw]
            output = np.multiply(input, np.fliplr(np.flipud(k)))  # flip
            res[i][j] = np.sum(output)
    return res  # implement the convolution with padding here


def processing(image, k):
    """
    :param image: 3D ndarray, RGB value
    :param k: kernel matrix
    :return: array to np.unit8
    """
    # TODO: blur the image, subtract the result to the input,
    #       add the result to the input, clip the values to the
    #       range [0,255] (remember warme-up exercise?), convert
    #       the array to np.unit8, and save the result

    shape = np.shape(image)
    res = np.zeros(shape, np.uint8)
    for i in range(0, 3):
        RGB = image[:, :, i]
        convolved = slow_convolve(RGB, k)
        temp = RGB + (RGB - convolved)
        res[:, :, i] = np.clip(temp, 0, 255)
    return res


if __name__ == '__main__':
    kernel1 = make_kernel(5, 2)  # todo: find better parameters
    kernel2 = make_kernel(10, 2)
    kernel3 = make_kernel(25, 5)
    # small sigma: no effect
    # large sigma: blurring
    # commonly used sigma: sigma = ksize / 5

    # TODO: chose the image you prefer
    img_origin = np.array(Image.open('guagua.jpg'))
    result1 = processing(img_origin, kernel1)
    result2 = processing(img_origin, kernel2)
    result3 = processing(img_origin, kernel3)

    plt.figure()

    plt.subplot(2, 2, 1)
    plt.imshow(img_origin)
    plt.title('Original')

    plt.subplot(2, 2, 2)
    plt.imshow(result1)
    plt.title('σ = 2 and kernel size 5', fontsize=10)

    plt.subplot(2, 2, 3)
    plt.imshow(result2)
    plt.title('σ = 2 and kernel size 10', fontsize=10)

    plt.subplot(2, 2, 4)
    plt.imshow(result3)
    plt.title('σ = 5 and kernel size 25', fontsize=10)

    plt.show()
