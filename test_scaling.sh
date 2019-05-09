for nproc in 1 4 9 18 36 72
do
  for npart in 2592
  do
    rm *npy
    echo "flock_numba_parallel $nproc $npart"
    python flock_numba_parallel.py $nproc $npart
    echo "flock_numba_joblib $nproc $npart"
    python flock_numba_joblib.py $nproc $npart
    echo "flock_numba_ray $nproc $npart"
    python flock_numba_ray.py $nproc $npart

    echo "predator_numba_parallel $nproc $npart"
    python predator_numba_parallel.py $nproc $npart
    echo "predator_numba_joblib $nproc $npart"
    python predator_numba_joblib.py $nproc $npart
    echo "predator_numba_ray $nproc $npart"
    python predator_numba_ray.py $nproc $npart
  done
done

rm *npy
python flock_numba.py 1 36
rm *npy
python flock_numba.py 4 144
rm *npy
python flock_numba.py 9 324
rm *npy
python flock_numba.py 18 648
rm *npy
python flock_numba.py 36 1296
rm *npy
python flock_numba.py 72 2592

rm *npy
python flock_numba_parallel.py 1 36
rm *npy
python flock_numba_parallel.py 4 144
rm *npy
python flock_numba_parallel.py 9 324
rm *npy
python flock_numba_parallel.py 18 648
rm *npy
python flock_numba_parallel.py 36 1296
rm *npy
python flock_numba_parallel.py 72 2592

rm *npy
python flock_numba_joblib.py 1 36
rm *npy
python flock_numba_joblib.py 4 144
rm *npy
python flock_numba_joblib.py 9 324
rm *npy
python flock_numba_joblib.py 18 648
rm *npy
python flock_numba_joblib.py 36 1296
rm *npy
python flock_numba_joblib.py 72 2592

rm *npy
python flock_numba_ray.py 1 36
rm *npy
python flock_numba_ray.py 4 144
rm *npy
python flock_numba_ray.py 9 324
rm *npy
python flock_numba_ray.py 18 648
rm *npy
python flock_numba_ray.py 36 1296
rm *npy
python flock_numba_ray.py 72 2592

rm *npy
python predator_numba.py 1 36
rm *npy
python predator_numba.py 4 144
rm *npy
python predator_numba.py 9 324
rm *npy
python predator_numba.py 18 648
rm *npy
python predator_numba.py 36 1296
rm *npy
python predator_numba.py 72 2592

rm *npy
python predator_numba_parallel.py 1 36
rm *npy
python predator_numba_parallel.py 4 144
rm *npy
python predator_numba_parallel.py 9 324
rm *npy
python predator_numba_parallel.py 18 648
rm *npy
python predator_numba_parallel.py 36 1296
rm *npy
python predator_numba_parallel.py 72 2592

rm *npy
python predator_numba_joblib.py 1 36
rm *npy
python predator_numba_joblib.py 4 144
rm *npy
python predator_numba_joblib.py 9 324
rm *npy
python predator_numba_joblib.py 18 648
rm *npy
python predator_numba_joblib.py 36 1296
rm *npy
python predator_numba_joblib.py 72 2592

rm *npy
python predator_numba_ray.py 1 36
rm *npy
python predator_numba_ray.py 4 144
rm *npy
python predator_numba_ray.py 9 324
rm *npy
python predator_numba_ray.py 18 648
rm *npy
python predator_numba_ray.py 36 1296
rm *npy
python predator_numba_ray.py 72 2592
