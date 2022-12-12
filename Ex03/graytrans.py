import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.transform import rotate
import scipy.fftpack as fft

import imutils


def DiscreteRadonTransform(image, steps):
    projections = []  ## Accumulate projections in a list.
    dTheta = -180.0 / steps  ## Angle increment for rotations.

    for i in range(steps):
        projections.append(rotate(image, i * dTheta).sum(axis=0))

    return np.vstack(projections)  # Return the projections as a sinogram


def fft_translate(projs):
    #Build 1-d FFTs of an array of projections, each projection 1 row of the array.
    return fft.rfft(projs, axis=1)


def ramp_filter(ffts):
    #Ramp filter a 2-d array of 1-d FFTs (1-d FFTs along the rows).
    ramp = np.floor(np.arange(0, ffts.shape[1]//2 + 0.1, 0.5))
    return ffts * ramp


def inverse_fft_translate(operator):
    return fft.irfft(operator, axis=1)


def back_project(operator):
    laminogram = np.zeros((operator.shape[1],operator.shape[1]))
    dTheta = 180.0 / operator.shape[0]
    for i in range(operator.shape[0]):
        temp = np.tile(operator[i],(operator.shape[1],1))
        temp = rotate(temp, dTheta*i)
        laminogram += temp
    return laminogram


if __name__ == '__main__':
    # image = cv2.imread("faulogoBG0.PNG", 0)
    image = cv2.imread("phantom128.jpg", cv2.IMREAD_GRAYSCALE)
    sinogram = DiscreteRadonTransform(image, len(image[0]))
    imutils.imshow(sinogram)
    # sinogram = imutils.imread('sinoLOGO.png')
    unfiltered_reconstruction = back_project(sinogram)
    imutils.imshow(unfiltered_reconstruction)

    frequency_domain_sinogram = fft_translate(sinogram)
    filtered_frequency_domain_sinogram = ramp_filter(frequency_domain_sinogram)
    filtered_spatial_domain_sinogram = inverse_fft_translate(filtered_frequency_domain_sinogram)
    imutils.imshow(filtered_spatial_domain_sinogram)

    reconstructed_image = back_project(filtered_spatial_domain_sinogram)
    imutils.imshow(reconstructed_image)


