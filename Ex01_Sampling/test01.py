import numpy as np
import matplotlib.pyplot as plt

samples = 200
k_max = 10000
frequency = 2
t = np.linspace(0, 1, samples)
omega = 2 * np.pi * frequency
temp = 0
for k in range(0, k_max, 1):
    ft = 8 * ((-1) ** k) * np.sin(omega * t * (2 * k + 1)) / ((2 * k + 1) ** 2) / (np. pi * np.pi)
    temp += ft
f = temp
plt.plot(t, f)
plt.show()
