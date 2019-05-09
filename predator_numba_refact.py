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
def inner_loop(y, arg, idx):
    a = arg[0]
    sx = y[idx * 2 + 0]
    sy = y[idx * 2 + 1]
    v = np.zeros((2,))
    for _idx in range(N):
        if idx != _idx:
            _sx = y[_idx * 2 + 0]
            _sy = y[_idx * 2 + 1]
            dsx = sx - _sx
            dsy = sy - _sy
            ds2 = dsx ** 2 + dsy ** 2
            v[0] += dsx* (1 / ds2 - a)
            v[1] += dsy* (1 / ds2 - a)
    return v

@jit
def outer_loop(y, arg, idx):
    b = arg[1]
    p = arg[2]
    szx = y[-2]
    szy = y[-1]
    sx = y[idx * 2 + 0]
    sy = y[idx * 2 + 1]
    v = inner_loop(y, arg, idx)
    dszx = sx - szx
    dszy = sy - szy
    dsz2 = dszx ** 2 + dszy ** 2
    v[0] = v[0] / N + b * dszx / dsz2
    v[1] = v[1] / N + b * dszy / dsz2
    vzx = dszx / dsz2 ** (p / 2)
    vzy = dszy / dsz2 ** (p / 2)
    return v, vzx, vzy

@jit
def outer_loop_batch(begin, end, y, arg):
    _vzx = 0
    _vzy = 0
    _dydt = np.zeros(((end-begin)*2,))
    for idx in range(end - begin):
        _v, __vzx, __vzy = outer_loop(y, arg, idx + begin)
        _dydt[idx*2:(idx+1)*2] = _v
        _vzx += __vzx
        _vzy += __vzy
    return _dydt, _vzx, _vzy

@jit
def f(t, y, arg):
    c = 0.4
    vzx = 0
    vzy = 0
    dydt = np.zeros((2*(N+1),))
    for rank in range(NPROC):
        begin = int(rank * N / NPROC)
        end = min(int((rank + 1) * N / NPROC), N)
        _dydt, _vzx, _vzy = outer_loop_batch(begin, end, y, arg)
        vzx += _vzx
        vzy += _vzy
        dydt[begin*2:end*2] = _dydt
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
