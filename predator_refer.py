import numpy as np
np.random.seed(204)
from scipy.integrate import ode
import matplotlib.pyplot as plt
import matplotlib
matplotlib
matplotlib.rc('font', family='FreeSans', size=14)


N = 400  # Number of preys
t0 = 0.0
y0= []
for _ in np.arange(N):
    theta = np.random.random() * 2 * np.pi
    radius = np.random.random()
    y0 += [radius * np.cos(theta),
           radius * np.sin(theta)]
y0 += [1 / np.sqrt(2), 1 / np.sqrt(2)]  # Predator initial condition
arg = [1, 0.2, 3]

def f(t, y, arg, c=0.4):
    a = arg[0]
    b = arg[1]
    p = arg[2]
    dydt = []
    szx = y[-2]
    szy = y[-1]
    vzx = 0
    vzy = 0
    for idx in range(N):
        sx = y[idx * 2 + 0]
        sy = y[idx * 2 + 1]
        vx = 0
        vy = 0
        for _idx in range(N):
            if idx != _idx:
                _sx = y[_idx * 2 + 0]
                _sy = y[_idx * 2 + 1]
                dsx = sx - _sx
                dsy = sy - _sy
                ds2 = dsx ** 2 + dsy ** 2
                vx += dsx* (1 / ds2 - a)
                vy += dsy* (1 / ds2 - a)
        dszx = sx - szx
        dszy = sy - szy
        dsz2 = dszx ** 2 + dszy ** 2
        vx = vx / N + b * dszx / dsz2
        vy = vy / N + b * dszy / dsz2
        vzx += dszx / dsz2 ** (p / 2)
        vzy += dszy / dsz2 ** (p / 2)
        dydt += [vx, vy]
    vzx *= c / N
    vzy *= c / N
    dydt += [vzx, vzy]
    return dydt

def plot(t, dt, y, y_prev, save=False):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(y[0:-2:2], y[1:-2:2], s=10, facecolors='k', edgecolors='k')
    ax.scatter(y[-2], y[-1], s=10, facecolors='r', edgecolors='r')
    for idx in range(N):
        ax.arrow(y[idx * 2 + 0], y[idx * 2 + 1], 
                 (y[idx * 2 + 0] - y_prev[idx * 2 + 0]) / dt / 5, 
                 (y[idx * 2 + 1] - y_prev[idx * 2 + 1]) / dt / 5,
                 head_width=0, head_length=0, fc='k', ec='k')
    ax.arrow(y[-2], y[-1], 
             (y[-2] - y_prev[-2]) / dt / 5, 
             (y[-1] - y_prev[-1]) / dt / 5,
             head_width=0, head_length=0, fc='r', ec='r')
    ax.set_xlim([-1.75, 1.25])
    ax.set_ylim([-1.75, 1.25])
    plt.tight_layout()
    if save:
        plt.savefig('debug/predator/%06d.png' % t, bbox_inches='tight', dpi=150)
    plt.close()

T = 40
dt = 0.01
t = t0 - T / 4
y = np.array(y0)
counter = 0
while t < T:
    y_prev = y.copy()
    if t > 0:
        c = 1.5
    else:
        c = 0.4
    y = y + dt * np.array(f(t, y, arg, c=c))
    t += dt
    counter += 1
    plot(counter // 1, dt, y, y_prev, save=(counter % 1 == 0))
