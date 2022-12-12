import numpy as np
import matplotlib.pyplot as plt

from test03 import sinus

t = np.linspace(0, 1, 200)
res = sinus(t)

plt.plot(t,res)
plt.show()