import numpy as np
import matplotlib.pyplot as plt
import os

filename = os.path.join('sounds', 'Piano.ff.A2.npy')
signal = np.load(filename)

# magnitude = np.abs(np.fft.fft(signal))
# n = signal.size
#
min_freq = 20
# time_step = 1 / 44100
# temp = np.fft.fftfreq(signal.size, time_step)
# freq = np.where(temp >= min_freq, temp, 0)
#
# print(freq)
fft_signal = np.fft.fft(signal)
magnitude = np.abs(fft_signal)
n = signal.size
fft_frequency = np.fft.fftfreq( n , 1 / 44100)
plt.plot(fft_frequency, fft_signal)
plt.show()
# # min_value = np.where(fft_frequency >= min_freq)
# min_value = np.where(fft_frequency >= min_freq, fft_frequency, 0)
# print(min_value)
# magnitude= magnitude[min_value[0][0] : len(min_value[0] + 1)]
# frequency = fft_frequency[np.argmax(magnitude)+min_value[0][0]]


# def load_sample(filename, duration, offset=4410):
#     # Complete this function
#     k = np.load(filename)
#     h = np.argmax(filename)
#     arr = k[h: h + offset + duration]
#     return arr
#
# sample = load_sample(os.path.join('sounds', 'Piano.ff.A2.npy'), duration=1291)
# print(sample[0])

# x = np.linspace(0, 1, 100)
# wave = np.sin(x)
# X = np.fft.fft(wave)
# pos = np.argmax(X)
# print(pos)
# print((X[pos]))
