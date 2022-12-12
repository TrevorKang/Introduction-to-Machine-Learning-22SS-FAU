# Xingjian Kang 23079135
# Jin Huang 23079099
# Lu Tian 23079258

"""Author: Kang"""

from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# ~ A2 109.5866767044307
# ~ A3 220.66463546809896
# ~ A4 440.75
# ~ A5 883.4999999999999
# ~ A6 1776.2664088616589
# ~ A7 3610.2512476828747
# ~ D6 1179.8780167508098


def load_sample(filename, duration = 4*44100, offset = 4410):
    # Complete this function
    signal = np.load(filename)
    # location of the highest peek and start of the signal
    h = np.argmax(np.abs(signal))
    return signal[h + offset: h + offset + duration]

def compute_frequency(signal, min_freq=20):
    # Complete this function
    # Compute the magnitude of the Fourier transform of the sample.
    # sampling frequency is 44100Hz.
    # Data Structure:
    #   2 arrays, mag for the vertical values of FFT, freq for the horizont
    #   linked with index
    fft_signal = np.fft.fft(signal)
    magnitude = np.abs(fft_signal)
    n = signal.size
    fft_frequency = np.fft.fftfreq(n, 1 / 44100)
    for i in range(0, magnitude.size):
        if fft_frequency[i] < min_freq:
            magnitude[i] = 0
    frequency = fft_frequency[np.argmax(magnitude)]
    return frequency


if __name__ == '__main__':
    # Implement the code to answer the questions here
    index = 0
    res = np.zeros(6)
    ev = [110, 220, 440, 880, 1760, 3520]
    # Tell the difference between expected value and real value
    for f in ('Piano.ff.A2.npy', 'Piano.ff.A3.npy', 'Piano.ff.A4.npy', 'Piano.ff.A5.npy', 'Piano.ff.A6.npy', 'Piano.ff.A7.npy'):
        sample = load_sample(os.path.join('sounds', f))
        res[index] = compute_frequency(sample)
        d = np.abs(res[index] - ev[index])
        print(f'The frequency of A{index + 2} key differs from the expect value with {d} Hz.')
        index += 1

    XX = load_sample(os.path.join('sounds', 'Piano.ff.XX.npy'))
    print(f'The missing note should be: {compute_frequency(XX)} Hz, means D6.')
    temp = input('Which on do you prefer? Cat? Dog? ')
    if temp == 'Cat':
        print('You made a right chose!')
    else:
        print('Seriously?')
    im = plt.imread('meme.jpg')
    plt.imshow(im)
    plt.show()

# This will be helpful:
# https://en.wikipedia.org/wiki/Piano_key_frequencies
