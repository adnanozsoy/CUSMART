
#include "aoso2_wrapper.h"
#include "algos/aoso2.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

#include "stdlib.h"

search_info average_shift_optimal_or_wrapper(search_parameters params)
{
    cuda_time kernel_time = {0}, total_time = {0};
    unsigned char *d_text, *d_pattern;
    int *d_match;
    unsigned int grid_dim, block_dim;

    setup_timers(&kernel_time, &total_time);
    get_kernel_configuration(params, &grid_dim, &block_dim);

    unsigned int *h_B = (unsigned int*)malloc(SIGMA * sizeof(unsigned int));
    unsigned int *d_B;
    cudaMalloc((void**)&d_B, SIGMA * sizeof(unsigned int));

    // Setup: malloc > timer start > memset/memcpy
    wrapper_setup(params, &d_text, &d_pattern, &d_match);

    // Preprocessing
    TicTocTimer preprocess_timer_start = tic();
    int q = 2;

    int p_len = params.pattern_size > 32 ? 32 : params.pattern_size;
    for (int i = 0; i < SIGMA; ++i)
        h_B[i] = ~0;
    unsigned int h = 0;
    unsigned int mm = 0;
    for (int j = 0; j < q; ++j) {
        for (int i = 0; i < p_len/q; ++i) {
            h_B[params.pattern[i*q+j]] &= ~(1<<h);
            ++h;
        }
        mm |= (1<<(h-1));
    }
    double preprocess_duration = toc(&preprocess_timer_start) * 1000;

    gpuErrchk( cudaMemcpy(d_B, h_B,
                          SIGMA * sizeof(unsigned int), cudaMemcpyHostToDevice) );

    // Kernel run
    gpuErrchk( cudaEventRecord(kernel_time.start) );
    if (params.pattern_size<=32) {
        average_shift_optimal_or<<<grid_dim, block_dim>>>(
            d_text, params.text_size, d_pattern, params.pattern_size,
            d_B, mm, params.stride_length, d_match);
    }
    else {
        average_shift_optimal_or_large<<<grid_dim, block_dim>>>(
            d_text, params.text_size, d_pattern, params.pattern_size,
            d_B, mm, params.stride_length, d_match);
    }
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaEventRecord(kernel_time.stop) );

    search_info timers = {0};
    timers.preprocess_duration = preprocess_duration;
    // Teardown: copy match back > timer stop > free
    wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
    // Free memory
    cudaFree(d_B);
    free(h_B);
    return timers;
}
