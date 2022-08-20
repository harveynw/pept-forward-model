import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle


def von_mises(theta_values, kappa):
    dens = np.exp(kappa*np.cos(theta_values))
    norm = np.sum(dens)
    return dens / norm


theta = np.linspace(-np.pi, np.pi, 1000)
cycol = cycle('bgrcmk')
for kappa in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
    plt.plot(theta,
             von_mises(theta_values=theta, kappa=kappa),
             c=next(cycol),
             label=rf'$\kappa={kappa}$')

ticks = [-1.5, -1, -0.5, 0.0, 0.5, 1, 1.5]
plt.xticks(np.array(ticks)*np.pi, [rf"${i}\pi$" for i in ticks])
plt.yticks([])
plt.legend()
plt.title(r'von Mises PDF for different values of $\kappa$')
plt.savefig('von_mises.eps', format='eps')
plt.show()


