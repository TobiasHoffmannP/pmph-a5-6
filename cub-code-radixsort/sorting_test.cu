//#include "../../cub-1.8.0/cub/cub.cuh"   // or equivalently <cub/device/device_histogram.cuh>
#include "cub/cub.cuh"
#include "helper.cu.h"
#include "bench-helper.h"

template<class Z>
bool validateZ(Z* A, uint32_t sizeAB) {
    for(uint32_t i = 1; i < sizeAB; i++)
      if (A[i-1] > A[i]){
        printf("INVALID RESULT for i:%d, (A[i-1]=%d > A[i]=%d)\n", i, A[i-1], A[i]);
        return false;
      }
    return true;
}

void randomInitNat(uint32_t* data, const uint32_t size, const uint32_t H) {
    for (int i = 0; i < size; ++i) {
        unsigned long int r = rand();
        data[i] = r % H;
    }
}

double sortRedByKeyCUB( uint32_t* data_keys_in
                      , uint32_t* data_keys_out
                      , const uint64_t N
) {
    int beg_bit = 0;
    int end_bit = 32;

    void * tmp_sort_mem = NULL;
    size_t tmp_sort_len = 0;

    { // sort prelude
        cub::DeviceRadixSort::SortKeys( tmp_sort_mem, tmp_sort_len
                                      , data_keys_in, data_keys_out
                                      , N,   beg_bit,  end_bit
                                      );
        cudaMalloc(&tmp_sort_mem, tmp_sort_len);
    }
    cudaCheckError();

    { // one dry run
        cub::DeviceRadixSort::SortKeys( tmp_sort_mem, tmp_sort_len
                                      , data_keys_in, data_keys_out
                                      , N,   beg_bit,  end_bit
                                      );
        cudaDeviceSynchronize();
    }
    cudaCheckError();

    // timing
    double elapsed;
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    float total_ms = 0.0f;

    for (int k = 0; k < GPU_RUNS; k++) {
        cudaEventRecord(ev_start);
        cub::DeviceRadixSort::SortKeys(tmp_sort_mem, tmp_sort_len,
                                    data_keys_in, data_keys_out,
                                    N, beg_bit, end_bit);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);

        float ms;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        total_ms += ms;
    }
    cudaCheckError();
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    elapsed = (total_ms * 1000.0) / GPU_RUNS;   // microseconds

    cudaFree(tmp_sort_mem);

    return elapsed;
}


int main (int argc, char * argv[]) {
    if (argc != 3) {
        printf("Usage: %s <size-of-array>\n", argv[0]);
        exit(1);
    }
    const uint64_t N = atoi(argv[1]);
    const uint64_t maxValue = atoi(argv[2]);

    //Use bench-helper to allocate and Initialize Host data with random values
    uint32_t* h_keys = createInput(N, maxValue);
    uint32_t* h_keys_res  = (uint32_t*) malloc(N*sizeof(uint32_t));

    //Allocate and Initialize Device data
    uint32_t* d_keys_in;
    uint32_t* d_keys_out;
    cudaSucceeded(cudaMalloc((void**) &d_keys_in,  N * sizeof(uint32_t)));
    cudaSucceeded(cudaMemcpy(d_keys_in, h_keys, N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    cudaSucceeded(cudaMalloc((void**) &d_keys_out, N * sizeof(uint32_t)));

    double elapsed = sortRedByKeyCUB( d_keys_in, d_keys_out, N );

    cudaMemcpy(h_keys_res, d_keys_out, N*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaCheckError();

    bool success = validateZ(h_keys_res, N);

    printf("CUB Sorting for N=%lu runs in: %.2f us, VALID: %d\n", N, elapsed, success);

    // Cleanup and closing
    cudaFree(d_keys_in); cudaFree(d_keys_out);
    free(h_keys); free(h_keys_res);

    return success ? 0 : 1;
}
