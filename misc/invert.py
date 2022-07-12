import matplotlib.pyplot as plt
import numpy as np

from geometry import Point

p = Point()
p.set_position_cylindrical(r=0.2, theta=np.random.uniform(0, 2*np.pi), z=0)

R = 0.25

fig, ax = plt.subplots()

x, y, z = p.get_position_cartesian()

phi = np.random.uniform(0, 2*np.pi)
print('Phi=', phi)

phi_formula_stuff = np.arccos((np.tan(phi)*x-y)/(R*np.sqrt(1+np.tan(phi)**2)))-np.arctan(1/np.tan(phi))
phi_2_a = + phi_formula_stuff
phi_2_b = - phi_formula_stuff
plt.scatter(x=[R*np.cos(phi_2_a)], y=[R*np.sin(phi_2_a)], c='red')
plt.scatter(x=[R*np.cos(phi_2_b)], y=[R*np.sin(phi_2_b)], c='red')

n = np.array([np.cos(phi), np.sin(phi)])
points = np.array([p.get_position_cartesian()[0:2] + i * n for i in np.linspace(-1, 1, 10)])
plt.plot(points[:,0], points[:,1])

plt.scatter(x=[x], y=[y])
ax.add_patch(plt.Circle((0, 0), radius=R, fill=False))

plt.xlim((-R, R))
plt.ylim((-R, R))


plt.show()
