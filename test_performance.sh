for nproc in 1 4 9 18 36 72
do
  for npart in 18 36 72 144 216 324 #648 1296
  do
    rm *npy
    echo "flock_numpy $nproc $npart"
    python flock_numpy.py $nproc $npart
    echo "flock_numba $nproc $npart"
    python flock_numba.py $nproc $npart
    echo "flock_numba_parallel $nproc $npart"
    python flock_numba_parallel.py $nproc $npart
    #echo "flock_numba_mproc $nproc $npart"
    #python flock_numba_mproc.py $nproc $npart
    echo "flock_numba_joblib $nproc $npart"
    python flock_numba_joblib.py $nproc $npart
    echo "flock_numba_ray $nproc $npart"
    python flock_numba_ray.py $nproc $npart
    echo "predator_numpy $nproc $npart"
    python predator_numpy.py $nproc $npart
    echo "predator_numba $nproc $npart"
    python predator_numba.py $nproc $npart
    echo "predator_numba_parallel $nproc $npart"
    python predator_numba_parallel.py $nproc $npart
    #echo "predator_numba_mproc $nproc $npart"
    #python predator_numba_mproc.py $nproc $npart
    echo "predator_numba_joblib $nproc $npart"
    python predator_numba_joblib.py $nproc $npart
    echo "predator_numba_ray $nproc $npart"
    python predator_numba_ray.py $nproc $npart
  done
done
