import numpy as np
np.random.seed(204)
from numba import jit
import time
import os
from multiprocessing import Pool
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
for theta_idx in np.arange(N):
    theta = (theta_idx + 1) / N * 2 * np.pi
    radius = 5 * np.sqrt(N)
    y0 += [radius * np.cos(theta),
            -np.sin(theta),
            radius * np.sin(theta),
            np.cos(theta)]
arg = np.asarray([0.07, 0.05, 10, 25, 10, 0.2])

@jit
def inner_loop(y, arg, idx):
    ca = arg[2]
    cr = arg[3]
    la = arg[4]
    lr = arg[5]
    sx = y[idx * 4 + 0]
    sy = y[idx * 4 + 2]
    u = np.zeros((2,))
    for _idx in range(N):
        if idx != _idx:
            _sx = y[_idx * 4 + 0]
            _sy = y[_idx * 4 + 2]
            dsx = sx - _sx
            dsy = sy - _sy
            ds = np.sqrt(dsx ** 2 + dsy ** 2)
            u[0] += -ca * np.exp(-ds / la) * (-dsx / ds / la) + \
                    cr * np.exp(-ds / lr) * (-dsx / ds / lr)
            u[1] += -ca * np.exp(-ds / la) * (-dsy / ds / la) + \
                    cr * np.exp(-ds / lr) * (-dsy / ds / lr)
    return u

@jit
def outer_loop(y, arg, idx):
    alpha = arg[0]
    beta = arg[1]
    _dydt = np.zeros((4,))
    _dydt[0] = y[idx * 4 + 1]
    _dydt[2] = y[idx * 4 + 3]
    v2 = _dydt[0] ** 2 + _dydt[2] **2
    u = inner_loop(y, arg, idx)
    _dydt[1] = (alpha - beta * v2) * _dydt[0] - u[0] / N
    _dydt[3] = (alpha - beta * v2) * _dydt[2] - u[1] / N
    return _dydt

@jit
def outer_loop_batch(rank, y, arg):
    begin = int(rank * N / NPROC)
    end = min(int((rank + 1) * N / NPROC), N)
    _dydt = np.zeros(((end-begin)*4,))
    for idx in range(end - begin):
        _dydt[idx*4:(idx+1)*4] = outer_loop(y, arg, idx + begin)
    return _dydt, begin, end

def f(t, y, arg):
    pool = Pool()
    pool_workers = [
        pool.apply_async(outer_loop_batch, (rank, y, arg,))
        for rank in range(NPROC)
    ]
    pool_segments = [worker.get(timeout=3) for worker in pool_workers]

    dydt = np.zeros((4*N,))
    for segment in pool_segments:
        _dydt, begin, end = segment
        dydt[begin*4:end*4] = _dydt

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
        if not os.path.isfile('flock_sol.npy'):
            np.save('flock_sol.npy', y)
        y_sol = np.load('flock_sol.npy')
        assert np.allclose(y, y_sol), 'The final solution is not correct!'
    print('Average trial time:', np.mean(trial_time))
    print('Trial time standard deviation:', np.std(trial_time))
