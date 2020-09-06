import matplotlib.pyplot as plt
import numpy as np


def Sigmoid(z_):
    return 1.0 / (1.0 + np.exp(-z_))


# Значения функции по x
z = np.arange(-7, 7, 0.1)
# Значения функции по y
phi_z = Sigmoid(z)
# График
plt.plot(z, phi_z)
# Вертикальная линия
plt.axvline(0.0, color='k')
# Границы y
plt.ylim(-0.1, 1.1)
# Label x
plt.xlabel('z')
# Label phi(z)
plt.ylabel('$\phi (z)$')
# Линии по y
plt.yticks([0.0, 0.5, 1.0])
# Оси
ax = plt.gca()
ax.yaxis.grid(True)

plt.show()