import numpy as np
np.random.seed(204)
from numba import jit
import time
import os
import psutil
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='Number of processors.')
parser.add_argument('nproc', nargs='?', type=int, default=4,
                    help='Number of parallel workers')
parser.add_argument('npart', nargs='?', type=int, default=50,
                    help='Number of parallel workers')
args = parser.parse_args()

NPROC = min(args.nproc, psutil.cpu_count())  # Run with at most max processors
N = max(args.npart, NPROC)  # Number of swarm particles
T = 1
dt = 0.01
t0 = 0.0
y0= []
for _ in np.arange(N):
    theta = np.random.random() * 2 * np.pi
    radius = np.random.random()
    y0 += [radius * np.cos(theta),
           radius * np.sin(theta)]
y0 += [1 / np.sqrt(2), 1 / np.sqrt(2)]  # Predator initial condition
arg = np.asarray([1, 0.2, 3])

@jit
def f(t, y, arg):
    a = arg[0]
    b = arg[1]
    c = 0.4
    p = arg[2]
    szx = y[-2]
    szy = y[-1]
    vzx = 0
    vzy = 0
    dydt = np.zeros((2*(N+1),))
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
        dydt[idx*2:(idx+1)*2] = [vx, vy]
    vzx *= c / N
    vzy *= c / N
    dydt[2*N:2*N+2] = [vzx, vzy]
    return dydt

if __name__ == '__main__':
    print('Running simulation with %d processors...' % NPROC)
    print('Running simulation with %d particles...' % N)
    trial = 0
    trial_time = []
    for trial in range(5):
        t = t0
        y = np.array(y0)
        print('Starting simulation trial %d...' % trial)
        exec_start = time.time()
        while t < T:
            y_prev = y.copy()
            y = y + dt * f(t, y, arg)
            t += dt
        exec_end = time.time()
        trial_time.append(exec_end - exec_start)
        print('Total execution time:', trial_time[-1])
        if not os.path.isfile('predator_sol.npy'):
            np.save('predator_sol.npy', y)
        y_sol = np.load('predator_sol.npy')
        assert np.allclose(y, y_sol), 'The final solution is not correct!'
    print('Average trial time:', np.mean(trial_time))
    print('Trial time standard deviation:', np.std(trial_time))
