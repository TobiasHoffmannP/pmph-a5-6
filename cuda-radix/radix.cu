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
#define ELM_PER_THREAD 22
#define BITS 8
#define TILE_SIZE (ELM_PER_THREAD*BLOCK_SIZE)
#define HIST_SIZE (1<<BITS)
#define ITERATIONS ((32+BITS-1)/BITS)
#define SHMEM_SIZE (HIST_SIZE*sizeof(unsigned int))
#define SHMEM_SIZE_SORT ((2*HIST_SIZE+TILE_SIZE)*sizeof(unsigned int))

#define TDIM 32
#define TROWS 8  

// in: rows x cols (row-major)  ->  out: cols x rows (row-major)
__global__ void transposeKernel(const unsigned int* in, unsigned int* out,
                                unsigned int rows, unsigned int cols) {
    __shared__ unsigned int tile[TDIM][TDIM + 1];   // +1: no bank conflicts on column access

    // read a 32x32 tile, coalesced along the input rows
    unsigned int x = blockIdx.x * TDIM + threadIdx.x;   // input column
    unsigned int y = blockIdx.y * TDIM + threadIdx.y;   // input row
    for (unsigned int k = 0; k < TDIM; k += TROWS)
        if (x < cols && y + k < rows)
            tile[threadIdx.y + k][threadIdx.x] = in[(y + k) * cols + x];
    __syncthreads();

    // write the tile transposed, coalesced along the output rows
    x = blockIdx.y * TDIM + threadIdx.x;                // output column (= input row)
    y = blockIdx.x * TDIM + threadIdx.y;                // output row    (= input column)
    for (unsigned int k = 0; k < TDIM; k += TROWS)
        if (x < rows && y + k < cols)
            out[(y + k) * rows + x] = tile[threadIdx.x][threadIdx.y + k];
}

__global__ void createHistogram(unsigned int* global_in, unsigned int* global_hist,
                                unsigned int i, unsigned int N, unsigned int num_blocks) {
    extern __shared__ unsigned int shmem[];
    unsigned int* hist = shmem;
    unsigned int tile_start = blockIdx.x * TILE_SIZE;

    // zero the histogram as gibberish could be in the shared memory
    for (unsigned int j = threadIdx.x; j < HIST_SIZE; j += blockDim.x)
        hist[j] = 0;

    __syncthreads();

    #pragma unroll
    for (unsigned int q = 0; q < ELM_PER_THREAD; q++) {
       unsigned int idx = tile_start + q * BLOCK_SIZE + threadIdx.x;
       if (idx < N) {
           // do direct coalecced reads from global memory.
           unsigned int segmented_bits = (global_in[idx] >> (i * BITS)) & (HIST_SIZE - 1);
           atomicAdd(&hist[segmented_bits], 1);
       }
    }
    __syncthreads();
    
    // copy histogram into global memory in column-major order to skip transposition
    for (unsigned int j = threadIdx.x; j < HIST_SIZE; j += blockDim.x)
        global_hist[blockIdx.x * HIST_SIZE + j] = hist[j];   // NOT j * num_blocks + blockIdx.x
}

__global__ void sortFromHistogram(unsigned int* global_in, unsigned int* global_out, unsigned int* global_hist,
          unsigned int* global_hist_scanned, unsigned int i,
				  unsigned int N, unsigned int num_blocks) {
    using BlockScan = cub::BlockScan<unsigned, BLOCK_SIZE>;
    __shared__ typename BlockScan::TempStorage scan_tmp;

    extern __shared__ unsigned int shmem[];
    unsigned int* tile = shmem;
    unsigned int* s_hist = shmem + TILE_SIZE;
    unsigned int* s_hist_scanned = shmem + TILE_SIZE + HIST_SIZE;
    unsigned int tile_start = blockIdx.x * TILE_SIZE;

    // copy from global to shared memory
    for (unsigned int j = threadIdx.x; j < TILE_SIZE; j += blockDim.x) {
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

        #pragma unroll                                 
        for (unsigned int q = 0; q < ELM_PER_THREAD; q++)
            key[q] = tile[threadIdx.x * ELM_PER_THREAD + q];

        __syncthreads();
    }
    // Step 3 in the last kernel from the hint:)
    // for (unsigned int j = threadIdx.x; j < HIST_SIZE; j += blockDim.x) {
    //     s_hist[j] = global_hist[j * num_blocks + blockIdx.x];
    //     s_hist_scanned[j] = global_hist_scanned[j * num_blocks + blockIdx.x];
    // }

    // both histograms are now block-major: row blockIdx.x, 256 consecutive words → coalesced
    for (unsigned int j = threadIdx.x; j < HIST_SIZE; j += blockDim.x) {
        s_hist[j]         = global_hist[blockIdx.x * HIST_SIZE + j];
        s_hist_scanned[j] = global_hist_scanned[blockIdx.x * HIST_SIZE + j];
    }

    __syncthreads();
    // Very simple sequential prefixsum. Does not speedup by
    // doing it in parallel.
    if (threadIdx.x == 0) {
        unsigned int sum = 0;
        for (unsigned int j = 0; j < HIST_SIZE; j++) {
            unsigned int v = s_hist[j];
            s_hist[j] = sum;
            sum += v;
        }
    }
    __syncthreads();

    unsigned int valid = min(TILE_SIZE, N - tile_start);

    #pragma unroll
    for (unsigned int q = 0; q < ELM_PER_THREAD; q++) {
        unsigned int idx = q * BLOCK_SIZE + threadIdx.x;
        if (idx >= valid) continue;
        unsigned int elm   = tile[idx];
        unsigned int digit = (elm >> (i * BITS)) & (HIST_SIZE - 1);
        unsigned int global_dest = s_hist_scanned[digit] + (idx - s_hist[digit]);
        global_out[global_dest] = elm;
    }    
}

int main(int argc, char** argv) {
    unsigned int N;
    unsigned int BOUND;
    
    { // reading the number of elements 
      if (argc != 3) { 
        printf("Num Args is: %d instead of 3. Exiting!\n", argc); 
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

    unsigned int NUM_BLOCKS = (N + TILE_SIZE - 1) / TILE_SIZE;
    unsigned int HIST_ELMS = NUM_BLOCKS*HIST_SIZE;
    size_t HIST_BYTES = HIST_ELMS*sizeof(unsigned int);
    
    // use the first CUDA device:
    cudaSetDevice(0);

    // allocate host memory
    unsigned int* h_in  = (unsigned int*)createInput(N, (unsigned long)BOUND);
    unsigned int* h_out = (unsigned int*) malloc(N*sizeof(unsigned int));

    unsigned int *d_in, *d_out, *d_hist, *d_hist_scanned, *d_hist_rows;
    cudaMalloc(&d_hist_rows, HIST_BYTES);
    cudaMalloc(&d_in,  N * sizeof(unsigned int));
    cudaMalloc(&d_out, N * sizeof(unsigned int));
    cudaMalloc(&d_hist,         HIST_BYTES);
    cudaMalloc(&d_hist_scanned, HIST_BYTES);

    // allocate CUB temp storage once, outside the timing
    void* d_temp = nullptr; size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_hist, d_hist_scanned, HIST_ELMS);
    cudaMalloc(&d_temp, temp_bytes);

    cudaMemcpy(d_in, h_in, N * sizeof(unsigned int), cudaMemcpyHostToDevice);
    dim3 tBlock(TDIM, TROWS);
    // ② input is NUM_BLOCKS x HIST_SIZE  → x over HIST_SIZE columns, y over NUM_BLOCKS rows
    dim3 tGridToDigit((HIST_SIZE  + TDIM - 1) / TDIM,
                      (NUM_BLOCKS + TDIM - 1) / TDIM);

    // ④ input is HIST_SIZE x NUM_BLOCKS  → x over NUM_BLOCKS columns, y over HIST_SIZE rows
    dim3 tGridToBlock((NUM_BLOCKS + TDIM - 1) / TDIM,
                      (HIST_SIZE  + TDIM - 1) / TDIM);
    // Dry run
    for (unsigned int i = 0; i < ITERATIONS; i++) {
        createHistogram<<<NUM_BLOCKS, BLOCK_SIZE, SHMEM_SIZE>>>(
            d_in, d_hist_rows, i, N, NUM_BLOCKS);                          // ① raw, block-major (kept)
        gpuAssert(cudaGetLastError());

        transposeKernel<<<tGridToDigit, tBlock>>>(
            d_hist_rows, d_hist, NUM_BLOCKS, HIST_SIZE);                   // ② → digit-major
        gpuAssert(cudaGetLastError());

        gpuAssert(cub::DeviceScan::ExclusiveSum(
            d_temp, temp_bytes, d_hist, d_hist_scanned, HIST_ELMS));       // ③ scan

        transposeKernel<<<tGridToBlock, tBlock>>>(
            d_hist_scanned, d_hist, HIST_SIZE, NUM_BLOCKS);                // ④ scanned → block-major
        gpuAssert(cudaGetLastError());

        sortFromHistogram<<<NUM_BLOCKS, BLOCK_SIZE, SHMEM_SIZE_SORT>>>(
            d_in, d_out, d_hist_rows, d_hist, i, N, NUM_BLOCKS);           // raw rows, scanned rows
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
        // restore data on device
        cudaMemcpy(d_in, h_in, N * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        cudaEventRecord(ev_start);
        for (unsigned int i = 0; i < ITERATIONS; i++) {
            createHistogram<<<NUM_BLOCKS, BLOCK_SIZE, SHMEM_SIZE>>>(
                d_in, d_hist_rows, i, N, NUM_BLOCKS);                          // ① raw, block-major (kept)
            gpuAssert(cudaGetLastError());

            transposeKernel<<<tGridToDigit, tBlock>>>(
                d_hist_rows, d_hist, NUM_BLOCKS, HIST_SIZE);                   // ② → digit-major
            gpuAssert(cudaGetLastError());

            gpuAssert(cub::DeviceScan::ExclusiveSum(
                d_temp, temp_bytes, d_hist, d_hist_scanned, HIST_ELMS));       // ③ scan

            transposeKernel<<<tGridToBlock, tBlock>>>(
                d_hist_scanned, d_hist, HIST_SIZE, NUM_BLOCKS);                // ④ scanned → block-major
            gpuAssert(cudaGetLastError());

            sortFromHistogram<<<NUM_BLOCKS, BLOCK_SIZE, SHMEM_SIZE_SORT>>>(
                d_in, d_out, d_hist_rows, d_hist, i, N, NUM_BLOCKS);           // raw rows, scanned rows
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
    cudaFree(d_out);
    cudaFree(d_hist);
    cudaFree(d_hist_scanned);
    cudaFree(d_temp);
}
