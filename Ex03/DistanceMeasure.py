'''

Created on 05.10.2016
Modified on 09.07.2022
@author: Xingjian KANG[ev00ykob], Jin HUANG[an46ykim] & Lu TIAN[ga94cice]

'''

import numpy as np
import matplotlib.pyplot as plt
# do not import more modules!


def calculate_R_Distance(Rx, Ry):
    '''
    calculate similarities of Ring features
    :param Rx: Ring features of Person X
    :param Ry: Ring features of Person Y
    :return: Similarity index of the two feature vectors
    '''
    # TODO: implement the Formula(8) in Chapter 4.1,
    k = len(Rx)
    res = 0
    for i in range(0, k):
        res += np.abs(Rx[i] - Ry[i])
    return res / k


def calculate_Theta_Distance(Thetax, Thetay):
    '''
    calculate similarities of Fan features
    :param Thetax: Fan features of Person X
    :param Thetay: Fan features of Person Y
    :return: Similiarity index of the two feature vectors
    '''
    # TODO: implement the Formula(9) ~ (12) in Chapter 4.4,
    thetaXi = sum(Thetax) / len(Thetax)
    thetaYi = sum(Thetay) / len(Thetay)

    # TODO: calculate lxx
    lxx = 0
    for i in range(0, len(Thetax)):
        lxx += (Thetax[i] - thetaXi) ** 2

    # TODO: calculate lyy
    lyy = 0
    for i in range(0, len(Thetay)):
        lyy += (Thetay[i] - thetaYi) ** 2

    # TODO: calculate lxy
    lxy = 0
    for i in range(0, len(Thetay)):
        lxy += (Thetax[i] - thetaXi) * (Thetay[i] - thetaYi)

    # TODO: calculate D
    D = 100 * (1 - lxy * lxy / (lxx * lyy))

    return D


