#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include "helper.h"

#define GPU_RUNS 100
#define UINT_SIZE_BITS sizeof(unsigned int)*8

__global__ void createHistogram(unsigned int* global_in, unsigned int* global_hist,
                                unsigned int i, unsigned int elm_per_thread,
                                unsigned int hist_size, unsigned int bits,
                                unsigned int tile_size, unsigned int N) {
    extern __shared__ unsigned int shmem[];
    unsigned int* hist = shmem;              
    unsigned int* tile = shmem + hist_size; 

    unsigned int tile_start = blockIdx.x * tile_size;

    // zero the histogram and load the tile cooperatively
    for (unsigned int j = threadIdx.x; j < hist_size; j += blockDim.x)
        hist[j] = 0;
    for (unsigned int j = threadIdx.x; j < tile_size; j += blockDim.x) {
        unsigned int g = tile_start + j;
        tile[j] = (g < N) ? global_in[g] : 0;
    }
    __syncthreads();

    for (unsigned int j = 0; j < elm_per_thread; j++) {
        unsigned int idx = threadIdx.x * elm_per_thread + j;
        if (tile_start + idx < N) {
            unsigned int elm = tile[idx];
            unsigned int segmented_bits = (elm >> (i * bits)) & (hist_size - 1);
            atomicAdd(&hist[segmented_bits], 1);
        }
    }
    __syncthreads();

    for (unsigned int j = threadIdx.x; j < hist_size; j += blockDim.x)
        global_hist[blockIdx.x * hist_size + j] = hist[j];
}

int main(int argc, char** argv) {
    unsigned int N;
    unsigned int BLOCK_SIZE;
    unsigned int BITS;
    unsigned int ELM_PER_THREAD;
    
    { // reading the number of elements 
      if (argc != 5) { 
        printf("Num Args is: %d instead of 4. Exiting!\n", argc); 
        exit(1);
      }

      N = atoi(argv[1]);
      printf("N is: %d\n", N);

      BLOCK_SIZE = atoi(argv[2]);
      printf("block size is: %d\n", BLOCK_SIZE);

      BITS = atoi(argv[3]);
      printf("number of bits is: %d\n", BITS);

      ELM_PER_THREAD = atoi(argv[4]);
      printf("number of elements per thread: %d\n", ELM_PER_THREAD);

      const unsigned int maxN = 500000000;
      if(N > maxN) {
          printf("N is too big; maximal value is %d. Exiting!\n", maxN);
          exit(2);
      }
    }

    unsigned int TILE_SIZE = ELM_PER_THREAD*BLOCK_SIZE;
    unsigned int NUM_BLOCKS = (unsigned int)ceil(N/TILE_SIZE);
    unsigned int HIST_SIZE = pow(2,BITS);
    unsigned int shmem_size = (HIST_SIZE + TILE_SIZE) * sizeof(unsigned int);
    // use the first CUDA device:
    cudaSetDevice(0);

    // allocate host memory
    unsigned int* h_in  = (unsigned int*) malloc(N*sizeof(unsigned int));
    unsigned int* h_out = (unsigned int*) malloc(NUM_BLOCKS*pow(2,BITS)*sizeof(unsigned int));

    // initialize the memory
    for(unsigned int i=0; i<N; ++i) {
        h_in[i] = i % 4;
    }

    // allocate device memory
    unsigned int* d_in;
    unsigned int* d_out;
    cudaMalloc((void**)&d_in, N*sizeof(unsigned int));
    cudaMalloc((void**)&d_out, NUM_BLOCKS*pow(2,BITS)*sizeof(unsigned int));

    // copy host memory to device
    cudaMemcpy(d_in, h_in, N*sizeof(unsigned int), cudaMemcpyHostToDevice);
    {
        double elapsed; struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        createHistogram<<<NUM_BLOCKS, BLOCK_SIZE, shmem_size>>>(d_in, d_out, 0, ELM_PER_THREAD, pow(2,BITS), BITS, TILE_SIZE, N);
        cudaDeviceSynchronize();
        
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (1.0 * (t_diff.tv_sec*1e6+t_diff.tv_usec)) / GPU_RUNS;
        double gigabytespersec = (2.0 * N * 4.0) / (elapsed * 1000.0);
        printf("The kernel took on average %f microseconds. GB/sec: %f \n", elapsed, gigabytespersec);
        
    }
        
    // check for errors
    gpuAssert( cudaPeekAtLastError() );

    // copy result from ddevice to host
    cudaMemcpy(h_out, d_out, NUM_BLOCKS*pow(2,BITS)*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    for (unsigned int b = 0; b < NUM_BLOCKS; b++) {
        for (unsigned int j = 0; j < HIST_SIZE; j++) {
                printf("result at block %d, bin %d: actual %d\n",
                       b, j, h_out[b * HIST_SIZE + j]);
        }
    }
    printf("Successful Validation.\n");

    // clean-up memory
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
}
