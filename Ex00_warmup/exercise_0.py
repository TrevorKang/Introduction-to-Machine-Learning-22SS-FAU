# 23079135, Xingjian Kang
# 23079099, Jin Huang
# 23079258, Lu Tian


import numpy as np
import matplotlib.pyplot as plt


# Complete this function such that returns the clipped values of the 1D
# numpy array passed as input to the given minimum and maximum. Example:
#    1 2 3 4 5 6 7 8
# clipped to a minimum of 3 and a maximum of 6 should give
#    3 3 3 4 5 6 6 6
# Note that the input array must not be modified.

# from numpy import double
#
# max = double(input("Input the maximum: "))
# min = double(input("Input the minimum: "))
#
# array = np.random.rand(100)
# print(array)
# res1 = np.where(array > min, array, min)
# print(res1)
# res2 = np.where(res1 < max, res1, max)
# print(res2)


def clip(array, minimum, maximum):
    res = np.where(array > minimum, array, minimum)
    return np.where(res < maximum, res, maximum)


if __name__ == '__main__':
    array = np.random.rand(100)
    result = clip(array, 0.2, 0.8)
    plt.plot(array, result, '.')
    plt.show()
