# Xingjian Kang 23079135
# Jin Huang 23079099
# Lu Tian 23079258

"""Author：Tian."""

import numpy as np

def createTriangleSignal(samples: int, frequency: int, k_max: int, ):
    t = np.linspace(0, 1, samples)
    ft = np.zeros_like(t)
    w=2*np.pi*frequency
    for i in range(len(t)):
        for k in range(0, k_max+1):
          ft[i] +=8*(pow(-1,k))*np.sin(w*(2*k+1)*t[i])/(pow((2*k+1),2))/(pow(np.pi,2))
    return ft
    # returns the signal as 1D-array (np.ndarray)
    # TODO
def createSquareSignal(samples: int, frequency: int, k_max: int):
    t = np.linspace(0,1,samples)
    ft = np.zeros_like(t)
    w = 2 * np.pi*frequency
    for i in range(len(t)):
        for k in range(1, k_max+1):
          ft[i] += 4*np.sin((2*k-1)*w*t[i])/(2*k-1)/np.pi
    return ft
    # returns the signal as 1D-array (np.ndarray)
    # TODO
def createSawtoothSignal(samples: int, frequency: int, k_max: int, amplitude: int):
    t = np.linspace(0, 1, samples)
    ft = np.zeros_like(t)
    w = 2 * np.pi * frequency
    A = amplitude
    for i in range(len(t)):
        for k in range(1, k_max+1):
         ft[i] +=((-1)*A/np.pi)*np.sin(w*t[i]*k)/k
         # returns the signal as 1D-array (np.ndarray)
    # TODO
    return ft+A/2

    # for k in range(1, k_max + 1, 1):
    #     ft = (-1) * A * np.sin(omega * t * k) / (np.pi * k)
    #     temp += ft
    # f = temp
    # return f + 0.5 * A
