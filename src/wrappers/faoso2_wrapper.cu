
#include "faoso2_wrapper.h"
#include "algos/faoso2.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"


#include <stdlib.h>

search_info fast_average_shift_optimal_or_wrapper(search_parameters params)
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
    int u = 2;

    int p_len = params.pattern_size > 32-u+1 ? 32-u+1 : params.pattern_size;
    unsigned int masq = 0;
    int mq = p_len/q;
    unsigned int h = mq;
    for (int j = 0; j < q; ++j) {
        masq |= (1<<h);
        masq |= (1<<h);
        h += mq;
        ++h;
    }
    for (int i = 0; i < SIGMA; ++i)
        h_B[i] = ~0;
    h = 0;
    unsigned int mm = 0;
    for (int j = 0; j < q; ++j) {
        for (int i = 0; i < mq; ++i) {
            h_B[params.pattern[i*q+j]] &= ~(1<<h);
            ++h;
        }
        mm |= (1<<(h-1));
        ++h;
        mm |= (1<<(h-1));
        ++h;
        --h;
    }
    double preprocess_duration = toc(&preprocess_timer_start) * 1000;

    gpuErrchk( cudaMemcpy(d_B, h_B,
                          SIGMA * sizeof(unsigned int), cudaMemcpyHostToDevice) );

    // Kernel run
    gpuErrchk( cudaEventRecord(kernel_time.start) );
    if (params.pattern_size<=32-u+1) {
        fast_average_shift_optimal_or<<<grid_dim, block_dim>>>(
            d_text, params.text_size, d_pattern, params.pattern_size,
            d_B, masq, mm, params.stride_length, d_match);
    }
    else {
        fast_average_shift_optimal_or_large<<<grid_dim, block_dim>>>(
            d_text, params.text_size, d_pattern, params.pattern_size,
            d_B, masq, mm, params.stride_length, d_match);
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
