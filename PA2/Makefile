all:
	gcc nbody.c -o nbody -lm -lX11 -pg
	gcc nbody-omp.c -o para -lm -lX11 -fopenmp
	gcc nbody-bht.c -o bht -lm -lX11 -fopenmp

clean:
	rm *.o *.out para nbody
