
#include "fsbndm_wrapper.h"
#include "algos/fsbndm.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

#include <stdlib.h>
#include <string.h>

search_info forward_simplified_backward_nondeterministic_dawg_matching_wrapper(search_parameters params){

    cuda_time kernel_time = {0}, total_time = {0};
    unsigned char *d_text, *d_pattern;
    int *d_match;
    unsigned int grid_dim, block_dim;

    setup_timers(&kernel_time, &total_time);
    get_kernel_configuration(params, &grid_dim, &block_dim);

    unsigned int *d_B;
    cudaMalloc((void**)&d_B, SIGMA * sizeof(unsigned int));
    unsigned int *h_B = (unsigned int*)malloc(SIGMA * sizeof(unsigned int));

    // Setup: malloc > timer start > memset/memcpy
    wrapper_setup(params, &d_text, &d_pattern, &d_match);


    /* Preprocessing */
    TicTocTimer preprocess_timer_start = tic();
    int p_len = params.pattern_size <= 32 ? params.pattern_size : 32;

    memset(h_B, 1, SIGMA * sizeof(unsigned int));
    for (int i = 0; i < p_len; ++i)
        h_B[params.pattern[i]] |= (1<<(p_len-i));

    double preprocess_duration = toc(&preprocess_timer_start) * 1000;
    
    cudaMemcpy(d_B, h_B, SIGMA * sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    // Kernel run
    gpuErrchk( cudaEventRecord(kernel_time.start) );
    if (params.pattern_size <= 32) {
        forward_simplified_backward_nondeterministic_dawg_matching<<<grid_dim, block_dim>>>(
            d_text, params.text_size, d_pattern, params.pattern_size, 
            d_B, params.stride_length, d_match);
    }
    else {
        forward_simplified_backward_nondeterministic_dawg_matching_large<<<grid_dim, block_dim>>>(
            d_text, params.text_size, d_pattern, params.pattern_size, 
            d_B, params.stride_length, d_match);
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
