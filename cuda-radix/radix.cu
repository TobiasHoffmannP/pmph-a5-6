#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "helper.h"
#include "../bench-helper.h"

#define GPU_RUNS 400
#define UINT_SIZE_BITS sizeof(unsigned int)*8
#define BLOCK_SIZE 256
#define ELM_PER_THREAD 23
#define BITS 8

__global__ void createHistogram(unsigned int* global_in, unsigned int* global_hist,
                                unsigned int i, unsigned int hist_size, unsigned int N,
                                unsigned int tile_size, unsigned int num_blocks) {
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

    for (unsigned int j = 0; j < ELM_PER_THREAD; j++) {
       unsigned int idx = threadIdx.x * ELM_PER_THREAD + j;
       if (tile_start + idx < N) {
           unsigned int elm = tile[idx];
           unsigned int segmented_bits = (elm >> (i * BITS)) & (hist_size - 1);
           atomicAdd(&hist[segmented_bits], 1);
       }
    }
    __syncthreads();
    
    // copy histogram into global memory in column-major order to skip transposition
    for (unsigned int j = threadIdx.x; j < hist_size; j += blockDim.x)
        global_hist[j * num_blocks + blockIdx.x] = hist[j];
}

__global__ void sortFromHistogram(unsigned int* global_in, unsigned int* global_out, unsigned int* global_hist,
          unsigned int* global_hist_scanned, unsigned int i,
				  unsigned int hist_size, unsigned int N,
				  unsigned int tile_size, unsigned int num_blocks) {
    using BlockScan = cub::BlockScan<unsigned, BLOCK_SIZE>;
    __shared__ typename BlockScan::TempStorage scan_tmp;

    extern __shared__ unsigned int shmem[];
    unsigned int* tile = shmem;
    unsigned int* s_hist = shmem + tile_size;
    unsigned int* s_hist_scanned = shmem + tile_size + hist_size;

    unsigned int tile_start = blockIdx.x * tile_size;

    // copy from global to shared memory
    for (unsigned int j = threadIdx.x; j < tile_size; j += blockDim.x) {
        unsigned int g = tile_start + j;
        tile[j] = (g < N) ? global_in[g] : 0xFFFFFFFFu; // assumes 32-bit integers
    }
    __syncthreads();

    // copy from shared to register memory
    unsigned int key[ELM_PER_THREAD];
    #pragma unroll
    for (unsigned int q = 0; q < ELM_PER_THREAD; q++)
        key[q] = tile[threadIdx.x * ELM_PER_THREAD + q];
    __syncthreads();
    
    // bit-wise sorting
    for (unsigned int j = 0; j < BITS; j++) {
        unsigned int bit_idx = i * BITS + j;
	
        unsigned int flag[ELM_PER_THREAD], prefix_sum[ELM_PER_THREAD];
	
        // generate flag array
        #pragma unroll
        for (unsigned int q = 0; q < ELM_PER_THREAD; q++) 
            flag[q] = ((key[q] >> bit_idx) & 1u) == 0; // predicate true if bit in position bit_idx is unset

        // exclusive scan to find prefix sum of flag array
        unsigned int numTrue;
        BlockScan(scan_tmp).ExclusiveSum(flag, prefix_sum, numTrue);
        __syncthreads();

        #pragma unroll
        for (unsigned int q = 0; q < ELM_PER_THREAD; q++) {
            unsigned int i = threadIdx.x * ELM_PER_THREAD + q;
            unsigned int dest = flag[q] ? prefix_sum[q] : numTrue + (i - prefix_sum[q]);
            tile[dest] = key[q];
        }
        __syncthreads();

        #pragma unroll                                   // (5) reload in new order
        for (unsigned int q = 0; q < ELM_PER_THREAD; q++)
            key[q] = tile[threadIdx.x * ELM_PER_THREAD + q];

        __syncthreads();
    }
    // TODO: Step 3 in the last kernel from the hint:)
    for (unsigned int j = threadIdx.x; j < hist_size; j += blockDim.x) {
        s_hist[j] = global_hist[j * num_blocks + blockIdx.x];
        s_hist_scanned[j] = global_hist_scanned[j * num_blocks + blockIdx.x];
    }
    // BlockScan(scan_tmp).ExclusiveSum(s_hist, s_hist);
    // __syncthreads();

    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int sum = 0;
        for (unsigned int j = 0; j < hist_size; j++) {
            unsigned int v = s_hist[j];
            s_hist[j] = sum;
            sum += v;
        }
    }
    __syncthreads();

    unsigned int valid = min(tile_size, N - tile_start);

    for (unsigned int q = 0; q < ELM_PER_THREAD; q++) {
        unsigned int idx = q * BLOCK_SIZE + threadIdx.x;
        if (idx >= valid) continue;
        unsigned int elm   = tile[idx];
        unsigned int digit = (elm >> (i * BITS)) & (hist_size - 1);
        unsigned int global_dest = s_hist_scanned[digit] + (idx - s_hist[digit]);
        global_out[global_dest] = elm;
    }    
}

int main(int argc, char** argv) {
    unsigned int N;
    unsigned int BOUND;
    
    { // reading the number of elements 
      if (argc != 3) { 
        printf("Num Args is: %d instead of 4. Exiting!\n", argc); 
        exit(1);
      }

      N = atoi(argv[1]);

      BOUND = atoi(argv[2]);

      const unsigned int maxN = 500000000;
      if(N > maxN) {
          printf("N is too big; maximal value is %d. Exiting!\n", maxN);
          exit(2);
      }
    }

    unsigned int TILE_SIZE = ELM_PER_THREAD*BLOCK_SIZE;
    unsigned int NUM_BLOCKS = (N + TILE_SIZE - 1) / TILE_SIZE;
    unsigned int HIST_SIZE = pow(2,BITS);
    unsigned int HIST_ELMS = NUM_BLOCKS*HIST_SIZE;
    size_t HIST_BYTES = HIST_ELMS*sizeof(unsigned int);
    unsigned int ITERATIONS = (unsigned int)ceil(UINT_SIZE_BITS/BITS);
    unsigned int shmem_size = (HIST_SIZE + TILE_SIZE) * sizeof(unsigned int);
    unsigned int shmem_size_sort = (2*HIST_SIZE + TILE_SIZE) * sizeof(unsigned int);

    //printf("TILE_SIZE: %d, NUM_BLOCK: %d, HIST_SIZE: %d, G_HIST_SIZE: %d, ITTERS: %d\n", TILE_SIZE, NUM_BLOCKS, HIST_SIZE, HIST_ELMS, ITERATIONS);
    // use the first CUDA device:
    cudaSetDevice(0);

    // allocate host memory
    unsigned int* h_in  = (unsigned int*)createInput(N, (unsigned long)BOUND);
    unsigned int* h_out = (unsigned int*) malloc(N*sizeof(unsigned int));

    unsigned int *d_in, *d_out, *d_hist, *d_hist_scanned;
    cudaMalloc(&d_in,  N * sizeof(unsigned int));
    cudaMalloc(&d_out, N * sizeof(unsigned int));
    cudaMalloc(&d_hist,         HIST_BYTES);
    cudaMalloc(&d_hist_scanned, HIST_BYTES);

    // allocate CUB temp storage once, outside the timing
    void* d_temp = nullptr; size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_hist, d_hist_scanned, HIST_ELMS);
    cudaMalloc(&d_temp, temp_bytes);

    cudaMemcpy(d_in, h_in, N * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Dry run
    for (unsigned int i = 0; i < ITERATIONS; i++) {
        createHistogram<<<NUM_BLOCKS, BLOCK_SIZE, shmem_size>>>(
            d_in, d_hist, i, HIST_SIZE, N, TILE_SIZE, NUM_BLOCKS);
        gpuAssert(cudaGetLastError());

        cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_hist, d_hist_scanned, HIST_ELMS);

        sortFromHistogram<<<NUM_BLOCKS, BLOCK_SIZE, shmem_size_sort>>>(
            d_in, d_out, d_hist, d_hist_scanned, i, HIST_SIZE, N, TILE_SIZE, NUM_BLOCKS);
        gpuAssert(cudaGetLastError());

        std::swap(d_in, d_out);
    }
    cudaDeviceSynchronize();

    
    double elapsed;
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    float total_ms = 0.0f;

    for (int k = 0; k < GPU_RUNS; k++) {
        // restore random input (not timed)
        cudaMemcpy(d_in, h_in, N * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        cudaEventRecord(ev_start);
        for (unsigned int i = 0; i < ITERATIONS; i++) {
            createHistogram<<<NUM_BLOCKS, BLOCK_SIZE, shmem_size>>>(
                d_in, d_hist, i, HIST_SIZE, N, TILE_SIZE, NUM_BLOCKS);
            gpuAssert(cudaGetLastError());

            cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_hist, d_hist_scanned, HIST_ELMS);

            sortFromHistogram<<<NUM_BLOCKS, BLOCK_SIZE, shmem_size_sort>>>(
                d_in, d_out, d_hist, d_hist_scanned, i, HIST_SIZE, N, TILE_SIZE, NUM_BLOCKS);
            gpuAssert(cudaGetLastError());

            std::swap(d_in, d_out);
        }
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);

        float ms;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        total_ms += ms;
    }

    cudaMemcpy(h_out, d_in, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    // Check validation, if invalid program exits with an error message
    for (unsigned int b = 1; b < N; b++) {
        if (h_out[b-1] > h_out[b]) {
            printf("INVALID RESULT for i:%d, (A[i-1]=%u > A[i]=%u)\n", b, h_out[b-1], h_out[b]);
            return 1;
        }
    }

    elapsed = (total_ms * 1000.0) / GPU_RUNS;   // microseconds per run
    double gigabytespersec = (2.0 * N * 4.0) / (elapsed * 1000.0);
    printf("CUDA Sorting for N=%u, runs in: %f us, GB/sec: %f, VALID: 1\n", N, elapsed, gigabytespersec);
        
    // check for errors
    gpuAssert( cudaPeekAtLastError() );

    // clean-up memory
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_hist);
    cudaFree(d_hist_scanned);
    cudaFree(d_temp);
}
