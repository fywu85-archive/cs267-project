import time
from numba import jit, prange
from multiprocessing import Pool
import ray
from joblib import Parallel, delayed

def sleep():
    time.sleep(1)

@jit(parallel=True)
def numba_sleep5():
    for _ in prange(5):
        time.sleep(1)

@ray.remote
def ray_sleep():
    time.sleep(1)

exec_start = time.time()
for _ in range(5):
    sleep()
exec_end = time.time()
print('Serial execution:', exec_end - exec_start)

numba_sleep5()
exec_start = time.time()
numba_sleep5()
exec_end = time.time()
print('numba parallel execution:', exec_end - exec_start)

exec_start = time.time()
pool = Pool(processes=5)
multiple_results = [pool.apply_async(sleep, ()) for idx in range(5)]
[res.get(timeout=3) for res in multiple_results]
exec_end = time.time()
print('multiprocessing execution:', exec_end - exec_start)

ray.init(num_cpus=5)
exec_start = time.time()
ray.get([ray_sleep.remote() for idx in range(5)])
exec_end = time.time()
print('ray execution:', exec_end - exec_start)

exec_start = time.time()
Parallel(n_jobs=5)(delayed(sleep)() for idx in range(5))
exec_end = time.time()
print('joblib execution:', exec_end - exec_start)
