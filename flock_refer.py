import numpy as np
np.random.seed(204)
from scipy.integrate import ode
import matplotlib.pyplot as plt
import matplotlib
matplotlib
matplotlib.rc('font', family='FreeSans', size=14)

N = 36  # Number of swarm particles
t0 = 0.0
y0= []
for theta_idx in np.arange(12):
    theta = (theta_idx + 1) / 12 * 2 * np.pi
    radius = 5
    y0 += [radius * np.cos(theta),
            -np.sin(theta),
            radius * np.sin(theta),
            np.cos(theta)]
for theta_idx in np.arange(24):
    theta = (theta_idx + 1) / 24 * 2 * np.pi
    radius = 10
    y0 += [radius * np.cos(theta),
            -np.sin(theta),
            radius * np.sin(theta),
            np.cos(theta)]
arg = [0.07, 0.05, 10, 25, 10, 0.2]

def f(t, y, arg):
    alpha = arg[0]
    beta = arg[1]
    ca = arg[2]
    cr = arg[3]
    la = arg[4]
    lr = arg[5]
    dydt = []
    for idx in range(N):
        sx = y[idx * 4 + 0]
        sy = y[idx * 4 + 2]
        vx = y[idx * 4 + 1]
        vy = y[idx * 4 + 3]
        v2 = vx ** 2 + vy **2
        ax = (alpha - beta * v2) * vx
        ay = (alpha - beta * v2) * vy
        ux = 0
        uy = 0
        for _idx in range(N):
            if idx != _idx:
                _sx = y[_idx * 4 + 0]
                _sy = y[_idx * 4 + 2]
                dsx = sx - _sx
                dsy = sy - _sy
                ds = np.sqrt(dsx ** 2 + dsy ** 2)
                ux += -ca * np.exp(-ds / la) * (-dsx / ds / la) + \
                      cr * np.exp(-ds / lr) * (-dsx / ds / lr)
                uy += -ca * np.exp(-ds / la) * (-dsy / ds / la) + \
                      cr * np.exp(-ds / lr) * (-dsy / ds / lr)
        ax -= ux / N
        ay -= uy / N
        dydt += [vx, ax, vy, ay]
    return dydt

def plot(t, dt, y, y_prev, save=False):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(y[0::2], y[1::2], s=10, facecolors='w', edgecolors='k')
    for idx in range(N):
        ax.arrow(y[idx * 2 + 0], y[idx * 2 + 1],
                 (y[idx * 2 + 0] - y_prev[idx * 2 + 0]) / dt / 2,
                 (y[idx * 2 + 1] - y_prev[idx * 2 + 1]) / dt / 2,
                 head_width=0.2, head_length=0.3, fc='k', ec='k')
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    #plt.show()
    plt.tight_layout()
    if save:
        plt.savefig('debug/flock/%06d.png' % t, bbox_inches='tight', dpi=150)
    plt.close()

T = 1000
dt = 0.01
t = t0
y = np.array(y0)
counter = 0
while t < T:
    y_prev = y.copy()
    y = y + dt * np.array(f(t, y, arg))
    t += dt
    counter += 1
    plot(counter // 1, dt, y[0::2], y_prev[0::2], save=(counter % 1 == 0))
