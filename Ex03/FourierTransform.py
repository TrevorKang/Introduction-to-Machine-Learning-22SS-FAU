'''

Created on 05.10.2016
Modified on 09.07.2022
@author: Xingjian KANG[ev00ykob], Jin HUANG[an46ykim] & Lu TIAN[ga94cice]

'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PalmprintAlignmentAutomatic import palmPrintAlignment
# do not import more modules!


def polarToKart(shape, r, theta):
    '''
    convert polar coordinates with origin in image center to kartesian
    :param shape: shape of the image
    :param r: radius from image center
    :param theta: angle
    :return: y, x
    '''
    # TODO: implement the Formula(5) in Chapter 3.3,
    yo = shape[0] // 2
    xo = shape[1] // 2  # image center

    y = r * np.sin(theta) + yo
    x = r * np.cos(theta) + xo

    return y, x


def calculateMagnitudeSpectrum(img) -> np.ndarray:
    '''
    use the fft to generate a magnitude spectrum and shift it to the image center.
    Hint: This can be done with numpy :)
    :param img: row binarized and aligned image as ndarray
    :return: magnitude, but convert into decibel!
    '''
    # TODO: FFT in 2D and then shift, then dB conversion,
    # np.fft.fft2: 2D discrete Fourier Transform
    # np.fft.fftshift: shift the zero-frequency component to the center of the spectrum
    shifted = np.fft.fftshift(np.fft.fft2(img))
    magnitude_dB = np.log10(np.abs(shifted)) * 20  # dB Conversion
    return magnitude_dB


def extractRingFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    '''
    Follow the approach to extract ring features
    :param magnitude_spectrum: 2D array with the magnitude dB value
    :param k: number of rings to extract = #features
    :param sampling_steps: times to sample one ring
    :return: feature vector of k features
    '''
    # TODO: implement the Formula(6) in Chapter 3.3,
    shape = np.shape(magnitude_spectrum)
    result = []
    for i in range(1, k + 1):
        sum: int = 0  # Ri
        for t in np.linspace(start=0, stop=np.pi, num=sampling_steps):
            for r in range(k * (i-1), k * i + 1):
                y, x = polarToKart(shape=shape, r=r, theta=t)
                sum += magnitude_spectrum[int(y), int(x)]  # The energy in each ring-like area
        result.append(sum)
    return np.array(result)


def extractFanFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    """
    Follow the approach to extract Fan features
    Assume all rays have same length regardless of angle.
    Their length should be set by the smallest feasible ray.
    :param magnitude_spectrum:
    :param k: number of fans-like features to extract
    :param sampling_steps: number of rays to sample from in one fan-like area
    :return: feature vector of length k
    """
    # TODO: implement the Formula(7) in Chapter 3.3,
    shape = np.shape(magnitude_spectrum)
    len = min(shape) # always in the width
    result = []
    for i in range(1, k + 1):
        sum: int = 0
        for t in np.linspace(start=i-1, stop=i, num=sampling_steps - 1):
            for r in range(0, len//2):
                y, x = polarToKart(shape=shape, r=r, theta=np.pi * t / k)
                sum += magnitude_spectrum[int(y), int(x)]
        result.append(sum)
    return np.array(result)


def calcuateFourierParameters(img, k, sampling_steps) -> (np.ndarray, np.ndarray):
    '''
    Extract Features in Fourier space following the paper.
    :param img: input image
    :param k: number of features to extract from each method
    :param sampling_steps: number of samples to accumulate for each feature
    :return: R, T feature vectors of length k
    '''
    # TODO: Ring-like and Fan-like Eigenvektors, all arr[k],
    magnitude_spectrum = calculateMagnitudeSpectrum(img)
    R = extractRingFeatures(magnitude_spectrum, k, sampling_steps)
    # print(len(R))
    T = extractFanFeatures(magnitude_spectrum, k, sampling_steps)
    # print(len(T))
    return R, T


if __name__ == '__main__':
    img = cv2.imread("guagua.jpg", cv2.IMREAD_GRAYSCALE)
    magnitude = calculateMagnitudeSpectrum(img)
    plt.subplot(1, 2, 1)
    plt.imshow(img, 'gray')
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude, 'gray')
    plt.show()
    # plt.subplots(2, 3)
    # arr = ['Hand1.jpg', 'Hand2.jpg', 'Hand3.jpg']
    # for i in range(1, 4):
    #     img = cv2.imread(arr[i - 1], cv2.IMREAD_GRAYSCALE)
    #     img_aligned = palmPrintAlignment(img)
    #     magnitude = calculateMagnitudeSpectrum(img_aligned)
    #     plt.subplot(2, 3, i)
    #     plt.imshow(img, 'gray')
    #     plt.subplot(2, 3, i+3)
    #     plt.imshow(magnitude, 'gray')
    # plt.show()


