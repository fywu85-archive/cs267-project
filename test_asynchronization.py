import numpy as np
import time
from multiprocessing import Pool
import ray
from joblib import Parallel, delayed

rand_data = np.random.random((5,))

def get_data(idx):
    time.sleep(0.2)
    return idx, rand_data[idx], idx/ 10

@ray.remote
def ray_get_data(idx):
    time.sleep(0.2)
    return idx, rand_data[idx], idx/ 10

exec_start = time.time()
counter = 0
data = np.zeros((5,))
for idx in range(5):
    _, _data, _counter = get_data(idx)
    data[idx] = _data
    counter += _counter
exec_end = time.time()
print('Serial execution:', exec_end - exec_start)
print('Generated data:', data)

exec_start = time.time()
counter = 0
data = np.zeros((5,))
pool = Pool(processes=5)
multiple_results = [pool.apply_async(get_data, (idx,)) for idx in range(5)]
segments = [res.get(timeout=3) for res in multiple_results]
for segment in segments:
    idx, _data, _counter = segment
    data[idx] = _data
    counter += _counter
exec_end = time.time()
print('multiprocessing execution:', exec_end - exec_start)
print('Generated data:', data)

ray.init(num_cpus=5)
exec_start = time.time()
counter = 0
data = np.zeros((5,))
segments = ray.get([ray_get_data.remote(idx) for idx in range(5)])
for segment in segments:
    idx, _data, _counter = segment
    data[idx] = _data
    counter += _counter
exec_end = time.time()
print('ray execution:', exec_end - exec_start)
print('Generated data:', data)

exec_start = time.time()
counter = 0
data = np.zeros((5,))
segments = Parallel(n_jobs=5)(delayed(get_data)(idx) for idx in range(5))
for segment in segments:
    idx, _data, _counter = segment
    data[idx] = _data
    counter += _counter
exec_end = time.time()
print('joblib execution:', exec_end - exec_start)
print('Generated data:', data)
