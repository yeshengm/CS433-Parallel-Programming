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

/* loop unrolled */
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

/* shared memory without bank conflict */
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



times in msec
 clock   self+sourced   self:  sourced script
 clock   elapsed:              other lines

000.003  000.003: --- VIM STARTING ---
000.076  000.073: Allocated generic buffers
000.107  000.031: locale set
000.121  000.014: GUI prepared
000.125  000.004: clipboard setup
000.129  000.004: window checked
000.465  000.336: inits 1
000.469  000.004: parsing arguments
000.470  000.001: expanding arguments
000.478  000.008: shell init
002.897  002.419: xsmp init
003.176  000.279: Termcap init
003.228  000.052: inits 2
003.369  000.141: init highlight
003.691  000.227  000.227: sourcing /usr/share/vim/vim74/debian.vim
004.014  000.184  000.184: sourcing /usr/share/vim/vim74/syntax/syncolor.vim
004.112  000.330  000.146: sourcing /usr/share/vim/vim74/syntax/synload.vim
026.463  000.811  000.811: sourcing /home/manifold/.vim/ftdetect/scala.vim
026.630  022.479  021.668: sourcing /usr/share/vim/vim74/filetype.vim
026.663  022.931  000.122: sourcing /usr/share/vim/vim74/syntax/syntax.vim
026.684  023.273  000.115: sourcing $VIM/vimrc
026.865  000.007  000.007: sourcing /usr/share/vim/vim74/filetype.vim
026.905  000.006  000.006: sourcing /usr/share/vim/vim74/filetype.vim
026.971  000.034  000.034: sourcing /usr/share/vim/vim74/ftplugin.vim
027.021  000.007  000.007: sourcing /usr/share/vim/vim74/filetype.vim
027.084  000.030  000.030: sourcing /usr/share/vim/vim74/indent.vim
027.359  000.214  000.214: sourcing /usr/share/vim/vim74/syntax/nosyntax.vim
027.599  000.155  000.155: sourcing /usr/share/vim/vim74/syntax/syncolor.vim
027.684  000.293  000.138: sourcing /usr/share/vim/vim74/syntax/synload.vim
027.710  000.591  000.084: sourcing /usr/share/vim/vim74/syntax/syntax.vim
028.069  000.161  000.161: sourcing /usr/share/vim/vim74/syntax/syncolor.vim
028.850  000.162  000.162: sourcing /usr/share/vim/vim74/syntax/syncolor.vim
029.118  000.154  000.154: sourcing /usr/share/vim/vim74/syntax/syncolor.vim
031.072  002.961  002.645: sourcing /home/manifold/.vim/colors/solarized.vim
031.662  000.219  000.219: sourcing /home/manifold/.vim/bundle/Vundle.vim/autoload/vundle.vim
031.877  000.146  000.146: sourcing /home/manifold/.vim/bundle/Vundle.vim/autoload/vundle/config.vim
032.780  006.075  001.913: sourcing $HOME/.vimrc
032.784  000.067: sourcing vimrc file(s)
033.106  000.226  000.226: sourcing /home/manifold/.vim/bundle/YouCompleteMe/plugin/youcompleteme.vim
033.380  000.066  000.066: sourcing /usr/share/vim/vim74/plugin/getscriptPlugin.vim
033.612  000.221  000.221: sourcing /usr/share/vim/vim74/plugin/gzip.vim
033.841  000.217  000.217: sourcing /usr/share/vim/vim74/plugin/logiPat.vim
034.023  000.170  000.170: sourcing /usr/share/vim/vim74/plugin/matchparen.vim
034.572  000.536  000.536: sourcing /usr/share/vim/vim74/plugin/netrwPlugin.vim
034.637  000.037  000.037: sourcing /usr/share/vim/vim74/plugin/rrhelper.vim
034.680  000.023  000.023: sourcing /usr/share/vim/vim74/plugin/spellfile.vim
034.844  000.150  000.150: sourcing /usr/share/vim/vim74/plugin/tarPlugin.vim
034.944  000.082  000.082: sourcing /usr/share/vim/vim74/plugin/tohtml.vim
035.122  000.156  000.156: sourcing /usr/share/vim/vim74/plugin/vimballPlugin.vim
035.324  000.177  000.177: sourcing /usr/share/vim/vim74/plugin/zipPlugin.vim
035.406  000.561: loading plugins
035.447  000.041: loading packages
036.154  000.707: inits 3
036.161  000.007: reading viminfo
037.928  001.767: setup clipboard
037.936  000.008: setting raw mode
037.944  000.008: start termcap
038.013  000.069: clearing screen
038.424  000.411: opening buffers
038.493  000.069: BufEnter autocommands
038.495  000.002: editing files in windows
066.236  027.599  027.599: sourcing /home/manifold/.vim/bundle/YouCompleteMe/autoload/youcompleteme.vim
216.925  150.831: VimEnter autocommands
216.929  000.004: before starting main loop
218.066  001.137: first screen update
218.069  000.003: --- VIM STARTED ---


times in msec
 clock   self+sourced   self:  sourced script
 clock   elapsed:              other lines

000.006  000.006: --- VIM STARTING ---
000.075  000.069: Allocated generic buffers
000.106  000.031: locale set
000.121  000.015: GUI prepared
000.125  000.004: clipboard setup
000.129  000.004: window checked
000.492  000.363: inits 1
000.497  000.005: parsing arguments
000.498  000.001: expanding arguments
000.507  000.009: shell init
002.260  001.753: xsmp init
002.516  000.256: Termcap init
002.566  000.050: inits 2
002.680  000.114: init highlight
003.029  000.276  000.276: sourcing /usr/share/vim/vim74/debian.vim
003.380  000.199  000.199: sourcing /usr/share/vim/vim74/syntax/syncolor.vim
003.489  000.361  000.162: sourcing /usr/share/vim/vim74/syntax/synload.vim
027.930  000.806  000.806: sourcing /home/manifold/.vim/ftdetect/scala.vim
028.072  024.542  023.736: sourcing /usr/share/vim/vim74/filetype.vim
028.106  025.032  000.129: sourcing /usr/share/vim/vim74/syntax/syntax.vim
028.126  025.417  000.109: sourcing $VIM/vimrc
028.329  000.007  000.007: sourcing /usr/share/vim/vim74/filetype.vim
028.369  000.006  000.006: sourcing /usr/share/vim/vim74/filetype.vim
028.434  000.034  000.034: sourcing /usr/share/vim/vim74/ftplugin.vim
028.473  000.006  000.006: sourcing /usr/share/vim/vim74/filetype.vim
028.579  000.029  000.029: sourcing /usr/share/vim/vim74/indent.vim
028.861  000.223  000.223: sourcing /usr/share/vim/vim74/syntax/nosyntax.vim
029.081  000.147  000.147: sourcing /usr/share/vim/vim74/syntax/syncolor.vim
029.168  000.275  000.128: sourcing /usr/share/vim/vim74/syntax/synload.vim
029.194  000.581  000.083: sourcing /usr/share/vim/vim74/syntax/syntax.vim
029.568  000.163  000.163: sourcing /usr/share/vim/vim74/syntax/syncolor.vim
030.352  000.161  000.161: sourcing /usr/share/vim/vim74/syntax/syncolor.vim
030.627  000.155  000.155: sourcing /usr/share/vim/vim74/syntax/syncolor.vim
032.594  002.983  002.667: sourcing /home/manifold/.vim/colors/solarized.vim
033.186  000.224  000.224: sourcing /home/manifold/.vim/bundle/Vundle.vim/autoload/vundle.vim
033.401  000.147  000.147: sourcing /home/manifold/.vim/bundle/Vundle.vim/autoload/vundle/config.vim
034.311  006.147  001.967: sourcing $HOME/.vimrc
034.315  000.071: sourcing vimrc file(s)
034.642  000.232  000.232: sourcing /home/manifold/.vim/bundle/YouCompleteMe/plugin/youcompleteme.vim
034.887  000.066  000.066: sourcing /usr/share/vim/vim74/plugin/getscriptPlugin.vim
035.123  000.222  000.222: sourcing /usr/share/vim/vim74/plugin/gzip.vim
035.354  000.219  000.219: sourcing /usr/share/vim/vim74/plugin/logiPat.vim
035.536  000.169  000.169: sourcing /usr/share/vim/vim74/plugin/matchparen.vim
036.088  000.539  000.539: sourcing /usr/share/vim/vim74/plugin/netrwPlugin.vim
036.154  000.037  000.037: sourcing /usr/share/vim/vim74/plugin/rrhelper.vim
036.199  000.022  000.022: sourcing /usr/share/vim/vim74/plugin/spellfile.vim
036.364  000.151  000.151: sourcing /usr/share/vim/vim74/plugin/tarPlugin.vim
036.465  000.082  000.082: sourcing /usr/share/vim/vim74/plugin/tohtml.vim
036.619  000.140  000.140: sourcing /usr/share/vim/vim74/plugin/vimballPlugin.vim
036.824  000.181  000.181: sourcing /usr/share/vim/vim74/plugin/zipPlugin.vim
036.905  000.530: loading plugins
036.952  000.047: loading packages
037.490  000.538: inits 3
037.499  000.009: reading viminfo
039.415  001.916: setup clipboard
039.424  000.009: setting raw mode
039.431  000.007: start termcap
039.495  000.064: clearing screen
040.010  000.515: opening buffers
040.081  000.071: BufEnter autocommands
040.083  000.002: editing files in windows
067.644  027.477  027.477: sourcing /home/manifold/.vim/bundle/YouCompleteMe/autoload/youcompleteme.vim
221.340  153.780: VimEnter autocommands
221.345  000.005: before starting main loop
222.610  001.265: first screen update
222.634  000.024: --- VIM STARTED ---
