import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib
matplotlib.rc('font', family='FreeSans', size=12)

data = []
with open('performance.txt', 'r') as file:
    for line in file:
        if 'flock_' in line or 'predator_' in line:
            _line = line.split(' ')
            data.append(int(_line[-2])) 
            data.append(int(_line[-1]))
        if 'Average' in line:
            _line = line.split(' ')
            data.append(float(_line[-1]))
        if 'standard' in line:
            _line = line.split(' ')
            data.append(float(_line[-1]))
data = np.asarray(data).reshape((-1, 5, 4))

flock_data = data[::2, ...].copy().reshape((8,6,5,4))
predator_data = data[1::2, ...].copy().reshape((8,6,5,4))

def remove_invalid(data):
    data[-1, 0] = np.nan
    data[-2, 0] = np.nan
    data[-1, 1] = np.nan
    return data

xlabels = [18, 36, 72, 144, 216, 324]
ylabels = [1, 2, 4, 8, 12, 18, 36, 72]
flock_numpy        = np.log10(flock_data[:,:,0,2])
flock_numba        = np.log10(flock_data[:,:,1,2])
flock_numba_par    = np.log10(flock_data[:,:,2,2])
flock_joblib       = np.log10(flock_data[:,:,3,2])
flock_ray          = np.log10(flock_data[:,:,4,2])
predator_numpy     = np.log10(predator_data[:,:,0,2])
predator_numba     = np.log10(predator_data[:,:,1,2])
predator_numba_par = np.log10(predator_data[:,:,2,2])
predator_joblib    = np.log10(predator_data[:,:,3,2])
predator_ray       = np.log10(predator_data[:,:,4,2])

data_arr = [
    flock_numpy, flock_numba, flock_numba_par, flock_joblib, flock_ray,
    predator_numpy, predator_numba, predator_numba_par, predator_joblib, 
    predator_ray
]
filename_arr = [
    'flock_numpy.png', 'flock_numba.png', 'flock_numba_par.png', 
    'flock_joblib.png', 'flock_ray.png', 'predator_numpy.png', 
    'predator_numba.png', 'predator_numba_par.png', 'predator_joblib.png', 
    'predator_ray.png'
]
for data, filename in zip(data_arr, filename_arr):
    fig = plt.figure(figsize=(4,3))
    plt.imshow(remove_invalid(data), cmap='RdYlGn_r', vmin=-2, vmax=2)
    cbar = plt.colorbar(ticks=np.arange(-2, 3, 1))
    cbar.ax.set_yticklabels(
        ['1e-2', '1e-1', '1e0', '1e1' , '1e2']
    )
    cbar.set_label('Running Time (s)')
    plt.xticks(np.arange(0, 6, 1))
    plt.yticks(np.arange(0, 8, 1))
    ax = plt.gca()
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)
    ax.xaxis.set_tick_params(rotation=45)
    plt.xlabel('Number of Particles')
    plt.ylabel('Number of Processors')
    plt.tight_layout()
    plt.savefig('figures/'+filename, bbox_inches='tight')
