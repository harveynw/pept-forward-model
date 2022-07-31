import numpy as onp
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random


def atan2(y, x):
    # return jnp.arctan2(y, x) + jnp.pi
    angle = jnp.arctan2(y, x)
    return angle if angle > 0.0 else 2 * jnp.pi + angle


def F_lambdas_hat(R, varphi, x, y):
    c_1 = x * jnp.cos(varphi) + y * jnp.sin(varphi)
    c_2 = jnp.sqrt(c_1 ** 2 - (x ** 2 + y ** 2 - R ** 2))
    return -c_1 + c_2, -c_1 - c_2


def F_phi_1(R, varphi, x, y):
    l_1, _ = F_lambdas_hat(R, varphi, x, y)
    return atan2(y + l_1 * jnp.cos(varphi), x + l_1 * jnp.sin(varphi))


def F_phi_2(R, varphi, x, y):
    _, l_2 = F_lambdas_hat(R, varphi, x, y)
    return atan2(y + l_2 * jnp.cos(varphi), x + l_2 * jnp.sin(varphi))


def F_z_1(R, varphi, theta, x, y, z):
    l_1, _ = F_lambdas_hat(R, varphi, x, y)
    return z + l_1 * jnp.cos(theta) / jnp.sin(theta)


def F_z_2(R, varphi, theta, x, y, z):
    _, l_2 = F_lambdas_hat(R, varphi, x, y)
    return z + l_2 * jnp.cos(theta) / jnp.sin(theta)


def G_phi(R, phi_1, x, y):
    c_x, c_y = x-R*jnp.cos(phi_1), y-R*jnp.sin(phi_1)
    omega = -2*R*(jnp.cos(phi_1)*c_x + jnp.sin(phi_1)*c_y)/(c_x**2 + c_y**2)
    return atan2(R*jnp.sin(phi_1)+omega*c_y, R*jnp.cos(phi_1)+omega*c_x)


def G_z(R, phi_1, z_1, x, y, z):
    c_x, c_y = x-R*jnp.cos(phi_1), y-R*jnp.sin(phi_1)
    omega = -2*R*(jnp.cos(phi_1)*c_x + jnp.sin(phi_1)*c_y)/(c_x**2 + c_y**2)
    return z_1 + omega*(z-z_1)


def jacobian_phi_1(R, phi_1, x, y):
    varphi = atan2(R*jnp.sin(phi_1) - y, R*jnp.cos(phi_1) - x)
    return grad(F_phi_1, 1)(R, varphi, x, y) # dF/dvarphi


def jacobian_z_1(R, phi_1, z_1, x, y, z):
    varphi = atan2(R*jnp.sin(phi_1) - y, R*jnp.cos(phi_1) - x)

    l_1 = jnp.sqrt((R*jnp.cos(phi_1)-x)**2 + (R*jnp.sin(phi_1)-y)**2 + (z_1 - z)**2)
    theta = jnp.arccos((z_1 - z)/l_1)

    return grad(F_z_1, 2)(R, varphi, theta, x, y, z)


def greater_than(x, threshold, gamma):
    return 0.5 * jnp.tanh((x - threshold) * gamma) + 0.5


def smaller_than(x, threshold, gamma):
    return 0.5 * jnp.tanh(-(x - threshold) * gamma) + 0.5


def detector_proj(R, min_phi, max_phi, min_z, max_z, x, y, z):
    # Clockwise
    p1 = (G_phi(R, max_phi, x, y), G_z(R, max_phi, max_z, x, y, z))
    p2 = (G_phi(R, max_phi, x, y), G_z(R, max_phi, min_z, x, y, z))

    p3 = (G_phi(R, min_phi, x, y), G_z(R, min_phi, min_z, x, y, z))
    p4 = (G_phi(R, min_phi, x, y), G_z(R, min_phi, max_z, x, y, z))

    return p1, p2, p3, p4




if __name__ == '__main__':
    # f = jit(jacobian_z_1)
    # print('Start')
    # for i in range(1000):
    #     f(0.5, jnp.pi, 0.1, 0.1 + 0.1 * i/1000, 0.1, 0.25)
    #     # jacobian_z_1(0.5, jnp.pi, 0.1, 0.1 + 0.1 * i/1000, 0.1, 0.25)

    for v in detector_proj(R=0.5, min_phi=jnp.pi-0.05, max_phi=jnp.pi+0.05, min_z=0.20, max_z=0.30, x=0.0, y=0.0, z=0.25):
        print(onp.array(v))



