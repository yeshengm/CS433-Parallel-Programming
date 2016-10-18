/* File:     mpi_io.c
 * Purpose:  Implement I/O functions that will be useful in an
 *           an MPI implementation of Dijkstra's algorithm.  
 *           In particular, the program creates an MPI_Datatype
 *           that can be used to implement input and output of
 *           a matrix that is distributed by block columns.  It
 *           also implements input and output functions that use
 *           this datatype.  Finally, it implements a function
 *           that prints out a process' submatrix as a string.
 *           This makes it more likely that printing the submatrix 
 *           assigned to one process will be printed without 
 *           interruption by another process.
 *
 * Compile:  mpicc -g -Wall -o mpi_io mpi_io.c
 * Run:      mpiexec -n <p> ./mpi_io (on lab machines)
 *           csmpiexec -n <p> ./mpi_io (on the penguin cluster)
 *
 * Input:    n:  the number of rows and the number of columns 
 *               in the matrix
 *           mat:  the matrix:  note that INFINITY should be
 *               input as 1000000
 * Output:   The submatrix assigned to each process and the
 *           complete matrix printed from process 0.  Both
 *           print "i" instead of 1000000 for infinity.
 *
 * Notes:
 * 1.  The number of processes, p, should evenly divide n.
 * 2.  You should free the MPI_Datatype object created by
 *     the program with a call to MPI_Type_free:  see the
 *     main function.
 * 3.  Example:  Suppose the matrix is
 *
 *        0 1 2 3
 *        4 0 5 6 
 *        7 8 0 9
 *        8 7 6 0
 *
 *     Then if there are two processes, the matrix will be
 *     distributed as follows:
 *
 *        Proc 0:  0 1    Proc 1:  2 3
 *                 4 0             5 6
 *                 7 8             0 9
 *                 8 7             6 0
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define MAX_STRING 10000
#define INFINITY 1000000

int Read_n(int my_rank, MPI_Comm comm);
MPI_Datatype Build_blk_col_type(int n, int loc_n);
void Read_matrix(int loc_mat[], int n, int loc_n, 
              MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm);
void Print_local_matrix(int loc_mat[], int n, int loc_n, int my_rank);
void Print_matrix(int loc_mat[], int n, int loc_n, 
              MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm);

void Dijkstra(int loc_mat[], int loc_dist[], int loc_pred[], int loc_n, int my_rank, int n);
void Print_dist(int dist[], int n);
void Print_path(int pred[], int dst, int n);

int main(int argc, char* argv[])
{
    int *loc_mat;
    int n, loc_n, p, my_rank;
    MPI_Comm comm;
    MPI_Datatype blk_col_mpi_t;
#  ifdef DEBUG
    int i, j;
#  endif

    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &my_rank);
  
    n = Read_n(my_rank, comm); // Read the size of matrix in thread 0
    loc_n = n/p;				  // local matrix and cols
    loc_mat = malloc(n*loc_n*sizeof(int));

#   ifdef DEBUG
    printf("Proc %d > p = %d, n = %d, loc_n = %d\n", my_rank, p, n, loc_n);
    /* This ensures that the matrix elements are initialized when */
    /* debugging.  It shouldn't be necessary */
    for (i = 0; i < n; i++)
        for (j = 0; j < loc_n; j++)
           loc_mat[i*loc_n + j] = -1;
#   endif   
   
    /* Build the special MPI_Datatype before doing matrix I/O */
    blk_col_mpi_t = Build_blk_col_type(n, loc_n);
    Read_matrix(loc_mat, n, loc_n, blk_col_mpi_t, my_rank, comm);
    // Print_local_matrix(loc_mat, n, loc_n, my_rank);

	// Print help info in master thread
    if (my_rank == 0) {
		printf("======= MPI Dijkstra =======\n");
		printf(" totol threads: %d\n", p);
    }
	

	int *loc_dist = malloc(loc_n * sizeof(int));
	int *loc_pred = malloc(loc_n * sizeof(int));
	
	Dijkstra(loc_mat, loc_dist, loc_pred, loc_n, my_rank, n);
	
	int *dist = malloc(n * sizeof(int));
	int *pred = malloc(n * sizeof(int));
	MPI_Gather(loc_dist, loc_n, MPI_INT, dist, loc_n, MPI_INT, 0, comm);
	MPI_Gather(loc_pred, loc_n, MPI_INT, pred, loc_n, MPI_INT, 0, comm);
	if (my_rank == 0) { Print_dist(dist, n); }
	if (my_rank == 0) {
		for (int i = 0; i < 9; ++i)
			Print_path(pred, i, n);
	}
    free(loc_mat);
	free(loc_dist);
	free(loc_pred);
    MPI_Type_free(&blk_col_mpi_t);
    MPI_Finalize();
    return 0;
}  /* main */

void Print_path(int pred[], int dst, int n)
{
	printf("path to %d: ", dst);
	int *stack = malloc(n*sizeof(int));
	int top = 0;
	
	while (dst != 0) {
		stack[top++] = dst;
		dst = pred[dst];
	}
	printf("0");
	while (--top >= 0)
		printf("->%d", stack[top]);
	printf("\n");
}

void Print_dist(int dist[], int n)
{
	printf("cost: ");
	for (int i = 0; i < n; ++i)
		printf("%d ", dist[i]);
	printf("\n");
}

void Find_min_loc_dist(int loc_dist[], int loc_known[], int loc_n, int my_min[], int my_rank)
{
	int loc_u = -1;
	int loc_min_dist = INFINITY;
	for (int loc_v = 0; loc_v < loc_n; loc_v++) {
		if (!loc_known[loc_v]) {
			if (loc_dist[loc_v] < loc_min_dist) {
				loc_u = loc_v;
				loc_min_dist = loc_dist[loc_v];
			}
		}
	}

	if (loc_u == -1) {
		my_min[0] = INFINITY;
		my_min[1] = 0;
		return;
	}
	my_min[0] = loc_dist[loc_u];
	my_min[1] = loc_u + my_rank*loc_n;
}

void Dijkstra(int loc_mat[], int loc_dist[], int loc_pred[], int loc_n, int my_rank, int n)
{
	int i, loc_u, loc_v, *loc_known;
	
	loc_known = malloc(loc_n * sizeof(int));
	for (loc_v = 0; loc_v < loc_n; loc_v++) {
		loc_dist[loc_v] = loc_mat[0*loc_n + loc_v];
		loc_known[loc_v] = 0;
		loc_pred[loc_v] = 0;
	}
	if (my_rank == 0)
		loc_known[0] = 1;
	
	for (int j = 1; j < n; j++) {
		int my_min[2], glbl_min[2];
		Find_min_loc_dist(loc_dist, loc_known, loc_n, my_min, my_rank);
		MPI_Allreduce(my_min, glbl_min, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);
		if (my_rank == glbl_min[1]/loc_n) {
			loc_known[glbl_min[1]%loc_n] = 1;
		}
		int new_loc_dist;
		for (loc_v = 0; loc_v < loc_n; loc_v++) {
			if (!loc_known[loc_v]) {
				new_loc_dist = glbl_min[0] + loc_mat[glbl_min[1]*loc_n + loc_v];	
				if (new_loc_dist < loc_dist[loc_v]) {
					loc_dist[loc_v] = new_loc_dist;
					loc_pred[loc_v] = glbl_min[1];
				}
			}
		}
	}
	free(loc_known);
}


/*---------------------------------------------------------------------
 * Function:  Read_n
 * Purpose:   Read in the number of rows in the matrix on process 0
 *            and broadcast this value to the other processes
 * In args:   my_rank:  the calling process' rank
 *            comm:  Communicator containing all calling processes
 * Ret val:   n:  the number of rows in the matrix
 */
int Read_n(int my_rank, MPI_Comm comm) {
   int n;
   // scanf only makes sense in thread 0
   if (my_rank == 0)
      scanf("%d", &n);
   MPI_Bcast(&n, 1, MPI_INT, 0, comm);
   return n;
}  /* Read_n */


/*---------------------------------------------------------------------
 * Function:  Build_blk_col_type
 * Purpose:   Build an MPI_Datatype that represents a block column of
 *            a matrix
 * In args:   n:  number of rows in the matrix and the block column
 *            loc_n = n/p:  number cols in the block column
 * Ret val:   blk_col_mpi_t:  MPI_Datatype that represents a block
 *            column
 */
MPI_Datatype Build_blk_col_type(int n, int loc_n) {
   MPI_Aint lb, extent;
   MPI_Datatype block_mpi_t;
   MPI_Datatype first_bc_mpi_t;
   MPI_Datatype blk_col_mpi_t;

   MPI_Type_contiguous(loc_n, MPI_INT, &block_mpi_t);
   MPI_Type_get_extent(block_mpi_t, &lb, &extent);

   MPI_Type_vector(n, loc_n, n, MPI_INT, &first_bc_mpi_t);
   MPI_Type_create_resized(first_bc_mpi_t, lb, extent,
         &blk_col_mpi_t);
   MPI_Type_commit(&blk_col_mpi_t);

   MPI_Type_free(&block_mpi_t);
   MPI_Type_free(&first_bc_mpi_t);

   return blk_col_mpi_t;
}  /* Build_blk_col_type */

/*---------------------------------------------------------------------
 * Function:  Read_matrix
 * Purpose:   Read in an nxn matrix of ints on process 0, and
 *            distribute it among the processes so that each
 *            process gets a block column with n rows and n/p
 *            columns
 * In args:   n:  the number of rows in the matrix and the submatrices
 *            loc_n = n/p:  the number of columns in the submatrices
 *            blk_col_mpi_t:  the MPI_Datatype used on process 0
 *            my_rank:  the caller's rank in comm
 *            comm:  Communicator consisting of all the processes
 * Out arg:   loc_mat:  the calling process' submatrix (needs to be 
 *               allocated by the caller)
 */
void Read_matrix(int loc_mat[], int n, int loc_n, 
      MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm) {
   int* mat = NULL, i, j;

   if (my_rank == 0) {
      mat = malloc(n*n*sizeof(int));
      for (i = 0; i < n; i++)
         for (j = 0; j < n; j++)
            scanf("%d", &mat[i*n + j]);
   }

   MPI_Scatter(mat, 1, blk_col_mpi_t,
           loc_mat, n*loc_n, MPI_INT, 0, comm);

   if (my_rank == 0) free(mat);
}  /* Read_matrix */


/*---------------------------------------------------------------------
 * Function:  Print_local_matrix
 * Purpose:   Store a process' submatrix as a string and print the
 *            string.  Printing as a string reduces the chance 
 *            that another process' output will interrupt the output.
 *            from the calling process.
 * In args:   loc_mat:  the calling process' submatrix
 *            n:  the number of rows in the submatrix
 *            loc_n:  the number of cols in the submatrix
 *            my_rank:  the calling process' rank
 */
void Print_local_matrix(int loc_mat[], int n, int loc_n, int my_rank) {
   char temp[MAX_STRING];
   char *cp = temp;
   int i, j;

   sprintf(cp, "Proc %d >\n", my_rank);
   cp = temp + strlen(temp);
   for (i = 0; i < n; i++) {
      for (j = 0; j < loc_n; j++) {
         if (loc_mat[i*loc_n + j] == INFINITY)
            sprintf(cp, " i ");
         else
            sprintf(cp, "%3d ", loc_mat[i*loc_n + j]);
         cp = temp + strlen(temp);
      }
      sprintf(cp, "\n");
      cp = temp + strlen(temp);
   }

   printf("%s\n", temp);
}  /* Print_local_matrix */


/*---------------------------------------------------------------------
 * Function:  Print_matrix
 * Purpose:   Print the matrix that's been distributed among the 
 *            processes.
 * In args:   loc_mat:  the calling process' submatrix
 *            n:  number of rows in the matrix and the submatrices
 *            loc_n:  the number of cols in the submatrix
 *            blk_col_mpi_t:  MPI_Datatype used on process 0 to
 *               receive a process' submatrix
 *            my_rank:  the calling process' rank
 *            comm:  Communicator consisting of all the processes
 */
void Print_matrix(int loc_mat[], int n, int loc_n,
      MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm) {
   int* mat = NULL, i, j;

   if (my_rank == 0) mat = malloc(n*n*sizeof(int));
   MPI_Gather(loc_mat, n*loc_n, MPI_INT,
         mat, 1, blk_col_mpi_t, 0, comm);
   if (my_rank == 0) {
      for (i = 0; i < n; i++) {
         for (j = 0; j < n; j++)
            if (mat[i*n + j] == INFINITY)
               printf(" i ");
            else
               printf("%3d ", mat[i*n + j]);
         printf("\n");
      }
      free(mat);
   }
}  /* Print_matrix */
