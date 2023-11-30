
#include "bmh_sbndm_wrapper.h"
#include "algos/bmh_sbndm.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

#include <stdlib.h>
#include <string.h>

search_info horspool_with_bndm_wrapper(
    search_parameters params)
{

    cuda_time kernel_time = {0}, total_time = {0};
    unsigned char *d_text, *d_pattern;
    int *d_match;
    unsigned int grid_dim, block_dim;

    setup_timers(&kernel_time, &total_time);
    get_kernel_configuration(params, &grid_dim, &block_dim);

    unsigned int* h_B = (unsigned int*) calloc(SIGMA, sizeof(unsigned int));
    unsigned int* d_B;
    cudaMalloc((void**)&d_B, SIGMA * sizeof(unsigned int));
    int* h_hbc = (int*) malloc(SIGMA * sizeof(int));
    int* d_hbc;
    cudaMalloc((void**)&d_hbc, SIGMA * sizeof(int));

    // Setup: malloc > timer start > memset/memcpy
    wrapper_setup(params, &d_text, &d_pattern, &d_match);

    int p_len = params.pattern_size <= 32 ? params.pattern_size : 32;
    /* Preprocessing */
    TicTocTimer preprocess_timer_start = tic();
    for (int i=0; i<p_len; i++)
        h_B[params.pattern[p_len-i-1]] |= (unsigned int)1<<(i+32-p_len);
    for (int i=0; i<SIGMA; i++) h_hbc[i]=p_len;
    for (int i=0; i<p_len; i++) h_hbc[params.pattern[i]]=p_len-i-1;
    unsigned int D = h_B[params.pattern[p_len-1]];
    int j=1;
    int shift=1;
    for (int i=p_len-2; i>0; i--, j++) {
        if (D & (1<<(p_len-1))) shift = j;
        D = (D<<1) & h_B[params.pattern[i]];
    }
    double preprocess_duration = toc(&preprocess_timer_start) * 1000;
    
    cudaMemcpy((d_text + params.text_size), params.pattern,
               params.pattern_size * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, SIGMA * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hbc, h_hbc, params.pattern_size * sizeof(int),
               cudaMemcpyHostToDevice);

    // Kernel run
    gpuErrchk( cudaEventRecord(kernel_time.start) );
    if (p_len <= 32) {
        horspool_with_bndm<<<grid_dim, block_dim>>>(
            d_text, params.text_size, d_pattern, params.pattern_size,
            d_B, d_hbc, shift, params.stride_length, d_match);
    }
    else {
        horspool_with_bndm_large<<<grid_dim, block_dim>>>(
            d_text, params.text_size, d_pattern, params.pattern_size,
            d_B, d_hbc, shift, params.stride_length, d_match);

    }
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaEventRecord(kernel_time.stop) );

    search_info timers = {0};
    timers.preprocess_duration = preprocess_duration;
    // Teardown: copy match back > timer stop > free
    wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
    // Free memory
    cudaFree(d_B);
    cudaFree(d_hbc);
    free(h_B);
    free(h_hbc);
    return timers;
}
