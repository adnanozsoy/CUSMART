
#include "bww_wrapper.h"
#include "algos/bww.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"


search_info bitparallel_wide_window_wrapper(search_parameters params)
{
    cuda_time kernel_time = {0}, total_time = {0};
    unsigned char *d_text, *d_pattern;
    int *d_match;
    unsigned int grid_dim, block_dim;

    setup_timers(&kernel_time, &total_time);
    get_kernel_configuration(params, &grid_dim, &block_dim);

    unsigned int *h_B = (unsigned int*)calloc(SIGMA, sizeof(unsigned int));
    unsigned int *h_C = (unsigned int*)calloc(SIGMA, sizeof(unsigned int));
    unsigned int *d_B, *d_C;
    cudaMalloc((void**)&d_B, SIGMA * sizeof(unsigned int));
    cudaMalloc((void**)&d_C, SIGMA * sizeof(unsigned int));

    // Setup: malloc > timer start > memset/memcpy
    wrapper_setup(params, &d_text, &d_pattern, &d_match);

    int p_len = params.pattern_size > 30 ? 30 : params.pattern_size;
    /* Preprocessing */
    TicTocTimer preprocess_timer_start = tic();
    /* Left to right automaton */
    unsigned int s = 1;
    for (int i = 0; i < p_len; ++i) {
        h_B[params.pattern[i]] |= s;
        s <<= 1;
    }
    s >>= 1;
    /* Right to left automaton */
    unsigned int t = 1;
    for (int i = p_len-1; i >= 0; --i) {
        h_C[params.pattern[i]] |= t;
        t <<= 1;
    }
    t >>= 1;
    double preprocess_duration = toc(&preprocess_timer_start) * 1000;
    
    cudaMemcpy(d_B, h_B, SIGMA * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, SIGMA * sizeof(unsigned int), cudaMemcpyHostToDevice);
    //Kernel run
    gpuErrchk( cudaEventRecord(kernel_time.start) );
    if (params.pattern_size <= 30) {
        bitparallel_wide_window<<<grid_dim, block_dim>>>(
            d_text, params.text_size, d_pattern, params.pattern_size,
            d_B, d_C, s, t, params.stride_length, d_match);
    }
    else {
        bitparallel_wide_window_large<<<grid_dim, block_dim>>>(
            d_text, params.text_size, d_pattern, params.pattern_size,
            d_B, d_C, s, t, params.stride_length, d_match);
    }
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaEventRecord(kernel_time.stop) );

    search_info timers = {0};
    timers.preprocess_duration = preprocess_duration;
    // Teardown: copy match back > timer stop > free
    wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
    free(h_B);
    free(h_C);
    cudaFree(d_B);
    cudaFree(d_C);
    return timers;
}
