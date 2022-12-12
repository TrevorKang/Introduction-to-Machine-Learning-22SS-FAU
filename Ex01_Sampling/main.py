import matplotlib.pyplot as plt
import numpy as np

from chirp import createChirpSignal
from decomposition import createTriangleSignal, createSquareSignal, createSawtoothSignal


# TODO: Test the functions imported in lines 1 and 2 of this file.

t = np.linspace(0, 1, 200)

# res1 = createChirpSignal(200, 1, 1, 10, True)
# plt.subplot(1,2,1)
# plt.title('Linear')
# plt.plot(t, res1)
# plt.xlabel('t (sec)')
# plt.grid(True)
#
# res2 = createChirpSignal(200, 1, 1, 10, False)
# plt.subplot(1,2,2)
# plt.title('Exponential')
# plt.plot(t, res2)
# plt.xlabel('t (sec)')
# plt.grid(True)
#
# plt.show()

res1 = createTriangleSignal(200, 2, 10000)
res2 = createSquareSignal(200, 2, 10000)
res3 = createSawtoothSignal(200, 2, 10000, 1)

plt.subplot(3, 1, 1)
plt.title('Triangle')
plt.plot(t, res1)
plt.xlabel('t (sec)')

plt.subplot(3, 1, 2)
plt.title('Square')
plt.plot(t, res2)
plt.xlabel('t (sec)')

plt.subplot(3, 1, 3)
plt.title('Sawtooth')
plt.plot(t, res3)
plt.xlabel('t (sec)')

plt.show()
