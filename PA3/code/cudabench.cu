/* matrix transpose program */
#include <stdio.h>

const int N = 1024;


/* naive CPU */
void naiveCPU()
{
}

/* naive GPU */
__global__ void matrixTranspose(float *_a, float *_b)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row
    int j = blockIdx.x * blockDim.x + threadIdx.x; // col

    int index_in = i * cols + j;
    int index_out = j * rows + i;

    b[index_out] = a[index_in];
}
