
#include "tndm_wrapper.h"
#include "algos/tndm.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

search_info two_way_nondeterministic_dawg_wrapper(search_parameters params){

    cuda_time kernel_time = {0}, total_time = {0};
    unsigned char *d_text, *d_pattern;
    int *d_match;
    unsigned int grid_dim, block_dim;

    setup_timers(&kernel_time, &total_time);
    get_kernel_configuration(params, &grid_dim, &block_dim);

    unsigned int *d_B, *d_restore;
    cudaMalloc((void**)&d_B, SIGMA * sizeof(unsigned int));
    cudaMalloc((void**)&d_restore, params.pattern_size+1 * sizeof(unsigned int));
    unsigned int *h_B = (unsigned int*)calloc(SIGMA, sizeof(unsigned int));
    unsigned int *h_restore = (unsigned int*)calloc(params.pattern_size+1, sizeof(unsigned int));

    // Setup: malloc > timer start > memset/memcpy
    wrapper_setup(params, &d_text, &d_pattern, &d_match);

    /* Preprocessing */
    TicTocTimer preprocess_timer_start = tic();
    unsigned int pat_len = params.pattern_size > 32 ? 32 : params.pattern_size;
    unsigned int s=1; 
    for (int i = pat_len-1; i>=0; i--){ 
        h_B[params.pattern[i]] |= s; 
        s <<= 1;
    }
    int last = pat_len;
    s = (unsigned int)(~0) >> (32-pat_len);
    for (int i=pat_len-1; i>=0; i--) {
        s &= h_B[params.pattern[i]]; 
        if (s & ((unsigned int)1<<(pat_len-1))) {
            if(i > 0)  last = i; 
        }
        h_restore[pat_len-i] = last;
        s <<= 1;
    }
    double preprocess_duration = toc(&preprocess_timer_start) * 1000;
    
    gpuErrchk( cudaMemcpy(d_B, h_B, SIGMA * sizeof(unsigned int), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_restore, h_restore, params.pattern_size+1 * sizeof(unsigned int), cudaMemcpyHostToDevice) );

    // Kernel run
    gpuErrchk( cudaEventRecord(kernel_time.start) );
    if(params.pattern_size <= 32){
        two_way_nondeterministic_dawg<<<grid_dim, block_dim>>>(
            d_text, params.text_size, d_pattern, params.pattern_size, 
            d_B, d_restore, params.stride_length, d_match);
    } else {
        two_way_nondeterministic_dawg_large<<<grid_dim, block_dim>>>(
            d_text, params.text_size, d_pattern, params.pattern_size, 
            d_B, d_restore, params.stride_length, d_match);
    }
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaEventRecord(kernel_time.stop) );

    search_info timers = {0};
    timers.preprocess_duration = preprocess_duration;
    // Teardown: copy match back > timer stop > free
    wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
    //Free Memory
    cudaFree(d_restore);
    cudaFree(d_B);
    return timers;
}
