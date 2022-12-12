# Xingjian Kang 23079135
# Jin Huang 23079099
# Lu Tian 23079258

"""Author: Huang"""
# Sampling Theorem
import numpy as np

def createChirpSignal(samplingrate: int, duration: int, freqfrom: int, freqto: int, linear: bool):
    # returns the chirp signal as list or 1D-array
    # TODO
    if linear:
        t = np.linspace(0, 1, samplingrate)
        f0 = freqfrom
        f1 = freqto
        T = duration
        c = (f1 - f0) / T
        y1 = np.sin(2 * np.pi * (f0 + c * 0.5 * t) * t)
        return y1

    else:

        t = np.linspace(0, 1, samplingrate)
        f0 = freqfrom
        f1 = freqto
        T = duration
        k = (f1 / f0) ** (1 / T)
        y1 = np.sin(2 * np.pi * f0 * (k ** t - 1) / np.log(k))
        return y1