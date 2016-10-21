# ./dijkstra < ./data/$1.txt
make
#data="/Users/changke/Desktop/data/${1}.txt"
data="/home/manifold/Desktop/dat/${1}.txt"
echo Running on $data with $2 threads
mpiexec -n $2 ./mpi_io $data

