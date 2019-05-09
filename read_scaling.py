import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib
matplotlib.rc('font', family='FreeSans', size=16)

data = []
with open('weak_scaling.txt', 'r') as file:
    for line in file:
        if 'Average' in line:
            _line = line.split(' ')
            data.append(float(_line[-1]))
        if 'standard' in line:
            _line = line.split(' ')
            data.append(float(_line[-1]))
data = np.asarray(data).reshape((-1, 6, 2))

nprocs = np.asarray([1, 4, 9, 18, 36, 72])
nparts = np.asarray([36, 144, 324, 648, 1296, 2592])
flock_weak_numba = data[0,:,0]
flock_weak_numba_par = data[1,:,0]
flock_weak_joblib = data[2,:,0]
flock_weak_ray = data[3,:,0]
predator_weak_numba = data[4,:,0]
predator_weak_numba_par = data[5,:,0]
predator_weak_joblib = data[6,:,0]
predator_weak_ray = data[7,:,0]

fig = plt.figure(figsize=(5,3.5))
ax = fig.gca()
for nproc, npart in zip(nprocs, nparts):
    ax.text(nproc, 0.5, 'N = %d' % npart, rotation=270)
    ax.axvline(x=nproc, linestyle='--', color='k')
ax.plot(nprocs, flock_weak_numba, '-o', label='flock numba')
ax.plot(nprocs, flock_weak_numba_par, '-*', label='flock numba par')
ax.plot(nprocs, flock_weak_joblib, '-+', label='flock joblib')
ax.plot(nprocs, flock_weak_ray, '-^', label='flock ray')
ax.set_xlabel('Number of Processors')
ax.set_ylabel('Running Time (s)')
ax.set_xscale('log')
ax.set_yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig('figures/'+'flock_weak_scale.png', bbox_inches='tight')

fig = plt.figure(figsize=(5,3.5))
ax = fig.gca()
for nproc, npart in zip(nprocs, nparts):
    ax.text(nproc, 0.5, 'N = %d' % npart, rotation=270)
    ax.axvline(x=nproc, linestyle='--', color='k')
ax.plot(nprocs, predator_weak_numba, '-o', label='predator numba')
ax.plot(nprocs, predator_weak_numba_par, '-*', label='predator numba par')
ax.plot(nprocs, predator_weak_joblib, '-+', label='predator joblib')
ax.plot(nprocs, predator_weak_ray, '-^', label='predator ray')
ax.set_xlabel('Number of Processors')
ax.set_ylabel('Running Time (s)')
ax.set_xscale('log')
ax.set_yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig('figures/'+'predator_weak_scale.png', bbox_inches='tight')

flock_strong_numba = np.asarray([flock_weak_numba[-1] for _ in range(len(nprocs))])
predator_strong_numba = np.asarray([predator_weak_numba[-1] for _ in range(len(nprocs))])
data = []
with open('strong_scaling.txt', 'r') as file:
    for line in file:
        if 'Average' in line:
            _line = line.split(' ')
            data.append(float(_line[-1]))
        if 'standard' in line:
            _line = line.split(' ')
            data.append(float(_line[-1]))
data = np.asarray(data).reshape((-1, 3, 2))

flock_strong = data[::2, ...].copy()
predator_strong = data[1::2, ...].copy()

nprocs = np.asarray([1, 4, 9, 18, 36, 72])
npart = 2592
flock_strong_numba_par = flock_strong[:, 0, 0]
flock_strong_joblib = flock_strong[:, 1, 0]
flock_strong_ray = flock_strong[:, 2, 0]
predator_strong_numba_par = predator_strong[:, 0, 0]
predator_strong_joblib = predator_strong[:, 1, 0]
predator_strong_ray = predator_strong[:, 2, 0]

fig = plt.figure(figsize=(5,3.5))
ax = fig.gca()
ax.plot(nprocs, flock_strong_numba, '-o', label='flock numba')
ax.plot(nprocs, flock_strong_numba_par, '-*', label='flock numba par')
ax.plot(nprocs, flock_strong_joblib, '-+', label='flock joblib')
ax.plot(nprocs, flock_strong_ray, '-^', label='flock ray')
ax.set_xlabel('Number of Processors')
ax.set_ylabel('Running Time (s)')
ax.set_xscale('log')
ax.set_yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig('figures/'+'flock_strong_scale.png', bbox_inches='tight')

fig = plt.figure(figsize=(5,3.5))
ax = fig.gca()
ax.plot(nprocs, predator_strong_numba, '-o', label='predator numba')
ax.plot(nprocs, predator_strong_numba_par, '-*', label='predator numba par')
ax.plot(nprocs, predator_strong_joblib, '-+', label='predator joblib')
ax.plot(nprocs, predator_strong_ray, '-^', label='predator ray')
ax.set_xlabel('Number of Processors')
ax.set_ylabel('Running Time (s)')
ax.set_xscale('log')
ax.set_yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig('figures/'+'predator_strong_scale.png', bbox_inches='tight')

