
#include "sbndmq2_wrapper.h"
#include "algos/sbndmq2.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

search_info simplified_backward_nondeterministic_dawg_qgram_wrapper(
    search_parameters params)
{

    cuda_time kernel_time = {0}, total_time = {0};
    unsigned char *d_text, *d_pattern;
    int *d_match;
    unsigned int grid_dim, block_dim;

    setup_timers(&kernel_time, &total_time);
    get_kernel_configuration(params, &grid_dim, &block_dim);

    unsigned int *d_B;
    cudaMalloc((void**)&d_B, SIGMA * sizeof(unsigned int));
    unsigned int *h_B = (unsigned int*)calloc(SIGMA, sizeof(unsigned int));

    // Setup: malloc > timer start > memset/memcpy
    wrapper_setup(params, &d_text, &d_pattern, &d_match);


    /* Preprocessing */
    TicTocTimer preprocess_timer_start = tic();
    int p_len = params.pattern_size <= 32 ? params.pattern_size : 32;

    int q = 2;
    int mMinusq = p_len - q +1;
    int mq = p_len - q;
    for (int i = 1; i <= p_len; ++i)
        h_B[params.pattern[p_len-i]] |= (1<<(i-1));

    unsigned int D = h_B[params.pattern[p_len-2]];
    int j = 1;
    int shift = 0;
    if (D & (1<<(p_len-1)))
        shift = p_len-j;
    for (int i = p_len-3; i>=0; i--) {
        D = (D<<1) & h_B[params.pattern[i]];
        j++;
        if (D & (1<<(p_len-1)))
            shift = p_len-j;
    }

    double preprocess_duration = toc(&preprocess_timer_start) * 1000;

    cudaMemcpy(d_B, h_B, SIGMA * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel run
    gpuErrchk( cudaEventRecord(kernel_time.start) );
    if (params.pattern_size <= 32) {
        simplified_backward_nondeterministic_dawg_qgram<<<grid_dim, block_dim>>>(
            d_text, params.text_size, d_pattern, params.pattern_size,
            d_B, mq, mMinusq, shift, params.stride_length, d_match);
    }
    else {
        simplified_backward_nondeterministic_dawg_qgram_large<<<grid_dim, block_dim>>>(
            d_text, params.text_size, d_pattern, params.pattern_size,
            d_B, mq, mMinusq, shift, params.stride_length, d_match);
    }
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaEventRecord(kernel_time.stop) );

    search_info timers = {0};
    timers.preprocess_duration = preprocess_duration;
    // Teardown: copy match back > timer stop > free
    wrapper_teardown(params, &timers, d_text, d_pattern, d_match);

    //Release Memory
    cudaFree(d_B);
    free(h_B);
    return timers;
}
