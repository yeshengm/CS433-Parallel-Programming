# ./dijkstra < ./data/$1.txt
make
mpiexec -n 2 ./mpi_io <./data/$1.txt

