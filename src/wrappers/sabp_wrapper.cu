
#include "sabp_wrapper.h"
#include "algos/sabp.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

search_info sabp_wrapper(search_parameters params) {

    cuda_time kernel_time = {0}, total_time = {0};
    unsigned char *d_text, *d_pattern;
    int *d_match;
    unsigned int grid_dim, block_dim;

    setup_timers(&kernel_time, &total_time);
    get_kernel_configuration(params, &grid_dim, &block_dim);

    unsigned int h_T[SIGMA];
    unsigned int* d_T;
    cudaMalloc((void**)&d_T, SIGMA * sizeof(unsigned int));
    unsigned int mask, mask2;
    int i;
    
    // Setup: malloc > timer start > memset/memcpy
    wrapper_setup(params, &d_text, &d_pattern, &d_match);

    int p_len = params.pattern_size <= 30 ? params.pattern_size : 30;

    /* Preprocessing */
    TicTocTimer preprocess_timer_start = tic();
    mask = 1;
    for (i = 1; i < p_len; ++i) mask = (mask << 1) | 1;
    for (i = 0; i < SIGMA; ++i) h_T[i] = mask;
    mask2 = 1;
    for (i = 0; i < p_len; ++i) {
      h_T[params.pattern[i]] &= ~mask2;
      mask2 <<= 1;
    }
    mask2 >>= 1;
    double preprocess_duration = toc(&preprocess_timer_start) * 1000;
    cudaMemcpy(d_T, h_T, SIGMA * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Kernel run
    gpuErrchk( cudaEventRecord(kernel_time.start) );
    if (params.pattern_size <= 30) {
        small_alphabet_bit_parallel<<<grid_dim, block_dim>>>(
            d_text, params.text_size, d_pattern, params.pattern_size,
            d_T, mask, mask2, params.stride_length, d_match);
    }
    else {
        small_alphabet_bit_parallel_large<<<grid_dim, block_dim>>>(
            d_text, params.text_size, d_pattern, params.pattern_size,
            d_T, mask, mask2, params.stride_length, d_match);

    }
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaEventRecord(kernel_time.stop) );

    search_info timers = {0};
    timers.preprocess_duration = preprocess_duration;
    // Teardown: copy match back > timer stop > free
    wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
    // Free memory
    cudaFree(d_T);
    return timers;
}
