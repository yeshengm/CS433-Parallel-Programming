/* matrix transpose program */
#include <stdio.h>

const int P = 32;

/* naive CPU */
void naiveCPU(float *src, float *dst, int M, int N)
{
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            dst[i*N+j] = src[j*N+i];
        }
    }
}

/* naive GPU */
__global__ void matrixTranspose(float *_a, float *_b, int cols,int rows)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row
    int j = blockIdx.x * blockDim.x + threadIdx.x; // col

    int index_in = i * cols + j;
    int index_out = j * rows + i;

    _b[index_out] = _a[index_in];
}


/* shared memory GPU */
__global__ void matrixTransposeShared(float *_a, float *_b, int cols, int rows)
{
    __shared__ float mat[P][P];
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    int i = by + threadIdx.y; int j = bx + threadIdx.x;
    int ti = bx + threadIdx.y; int tj = by + threadIdx.x;

    if (i < rows && j < cols)
        mat[threadIdx.x][threadIdx.y] = _a[i*cols + j];
    __syncthreads();
    if (tj < cols && ti < rows)
        _b[ti*rows+tj] = mat[threadIdx.y][threadIdx.x];
}

/* shared memory without bank conflict */
__global__ void matrixTransposeUnrolled(float *_a, float *_b, int cols, int rows)
{
    __shared__ float mat[P][P+1];
    int x = blockIdx.x * P + threadIdx.x;
    int y = blockIdx.y * P + threadIdx.y;

#pragma unroll
    for (int k = 0; k < P; k += 8) {
        if (x < rows && y+k < cols)
            mat[threadIdx.y+k][threadIdx.x] = _a[(y+k)*rows + x];
    }

    __syncthreads();

    x = blockIdx.y * P + threadIdx.x;
    y = blockIdx.x * P + threadIdx.y;
#pragma unroll
    for (int k = 0; k < P; k += 8) {
        if (x < cols && y+k < rows)
            _b[(y+k)*cols + x] = mat[threadIdx.x][threadIdx.y+k];
    }
}

/* loop unrolled */
__global__ void matrixTransposeSharedwBC(float *_a, float *_b, int cols, int rows)
{
    __shared__ float mat[P][P+1];
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    int i = by + threadIdx.y; int j = bx + threadIdx.x;
    int ti = bx + threadIdx.y; int tj = by + threadIdx.x;

    if (i < rows && j < cols)
        mat[threadIdx.x][threadIdx.y] = _a[i*cols + j];
    __syncthreads();
    if (tj < cols && ti < rows)
        _b[ti*rows+tj] = mat[threadIdx.y][threadIdx.x];
}


int main(int argc, char **argv)
{
    /* N*M matrix, parallelism is P */
    const int N = 1024;
    const int M = 1024;
    const int matSize = N * M * sizeof(float);
    dim3 gridDim(N/P, M/P, 1);
    dim3 blockDim(P , P, 1);

    /* configuration of GPU */
    printf("===================\n");
    printf("Matrix: %d * %d\n", N, M);
    printf("Grid:   %d * %d * %d\n", gridDim.x, gridDim.y, gridDim.z);
    printf("Block:  %d * %d * %d\n", blockDim.x, blockDim.y, blockDim.z);
    printf("===================\n");
    

    /* allocate memory for matrix in host */
    float *h_matrix = (float *) malloc(matSize);
    float *h_transpose = (float *) malloc(matSize);
    
    /* allocate memory for matrix in device */
    float *d_matrix, *d_transpose;
    cudaMalloc(&d_matrix, matSize);
    cudaMalloc(&d_transpose, matSize);

    /* randomly generate a matrix in host */
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            h_matrix[i*N+j] = (float)rand() / (float)(RAND_MAX) * 100.0;
        }
    }
    
    /* utility for recording start and finish time */
    cudaEvent_t tStart, tEnd;
    float duration;
    cudaEventCreate(&tStart);
    cudaEventCreate(&tEnd);
    const int nIterations = 100;


    /* 1. naive CPU transpose */
    cudaEventRecord(tStart, 0);
    for (int i = 0; i < nIterations; ++i) {
        naiveCPU(h_matrix, h_transpose, N, M);
    }
    cudaEventRecord(tEnd, 0);
    cudaEventSynchronize(tEnd); // waits for record to terminate
    cudaEventElapsedTime(&duration, tStart, tEnd);
    printf("\nNaive CPU: %f\n", duration / nIterations);


    /* 2. naive GPU transpose */
    cudaMemcpy(d_matrix, h_matrix, matSize, cudaMemcpyHostToDevice);
    cudaMemset(d_transpose, 0, matSize);
    cudaEventRecord(tStart, 0);
    for (int i = 0; i < nIterations; ++i) {
        matrixTranspose<<<gridDim,blockDim>>>(d_matrix, d_transpose, N, M);
    }
    cudaEventRecord(tEnd, 0);
    cudaEventSynchronize(tEnd);
    cudaEventElapsedTime(&duration, tStart, tEnd);
    cudaMemcpy(h_transpose, d_transpose, matSize, cudaMemcpyDeviceToHost);
    printf("\nNaive GPU: %f\n", duration / nIterations);
    
    /* 3. shared memory GPU transpose */
    cudaMemcpy(d_matrix, h_matrix, matSize, cudaMemcpyHostToDevice);
    cudaMemset(d_transpose, 0, matSize);
    cudaEventRecord(tStart, 0);
    for (int i = 0; i < nIterations; ++i) {
        matrixTransposeShared<<<gridDim,blockDim>>>(d_matrix, d_transpose, N, M);
    }
    cudaEventRecord(tEnd, 0);
    cudaEventSynchronize(tEnd);
    cudaEventElapsedTime(&duration, tStart, tEnd);
    cudaMemcpy(h_transpose, d_transpose, matSize, cudaMemcpyDeviceToHost);
    printf("\nShared GPU: %f\n", duration / nIterations);


    /* 4. shared memory GPU transpose without bank conflict */
    cudaMemcpy(d_matrix, h_matrix, matSize, cudaMemcpyHostToDevice);
    cudaMemset(d_transpose, 0, matSize);
    cudaEventRecord(tStart, 0);
    for (int i = 0; i < nIterations; ++i) {
        matrixTransposeSharedwBC<<<gridDim,blockDim>>>(d_matrix, d_transpose, N, M);
    }
    cudaEventRecord(tEnd, 0);
    cudaEventSynchronize(tEnd);
    cudaEventElapsedTime(&duration, tStart, tEnd);
    cudaMemcpy(h_transpose, d_transpose, matSize, cudaMemcpyDeviceToHost);
    printf("\nSharedwBC GPU: %f\n", duration / nIterations);


    duration = 0;
    /* 5. unrolled GPU transpose */
    dim3 blockDimUnroll(P, 8, 1);
    cudaMemcpy(d_matrix, h_matrix, matSize, cudaMemcpyHostToDevice);
    cudaMemset(d_transpose, 0, matSize);
    cudaEventRecord(tStart, 0);
    for (int i = 0; i < nIterations; ++i) {
        matrixTransposeUnrolled<<<gridDim,blockDimUnroll>>>(d_matrix, d_transpose, N, M);
    }
    cudaEventRecord(tEnd, 0);
    cudaEventSynchronize(tEnd);
    cudaEventElapsedTime(&duration, tStart, tEnd);
    cudaMemcpy(h_transpose, d_transpose, matSize, cudaMemcpyDeviceToHost);
    printf("\nUnrolled GPU: %f\n", duration / nIterations);
    
    return 0;
}

