
#include "sbndm2_wrapper.h"
#include "algos/sbndm2.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

search_info simplified_backward_nondeterministic_dawg_unrolled_wrapper(search_parameters params){

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

    for (int i = 1; i <= p_len; ++i) 
        h_B[params.pattern[p_len-i]] |= (1<<(i-1));

    int j = 1;
    int shift;
    unsigned int D; 
    if (params.pattern_size <=32) {
        D = h_B[params.pattern[params.pattern_size-2]]; 
        shift = 0;
        if(D & (1<<(params.pattern_size-1))) shift = params.pattern_size-j;
        for(int i=params.pattern_size-3; i>=0; i--) {
          D = (D<<1) & h_B[params.pattern[i]];
          j++;
          if(D & (1<<(params.pattern_size-1))) shift = params.pattern_size-j;
        }
    }
    else {
        D = h_B[params.pattern[p_len-1]]; 
        shift = 1;
        for(int i=p_len-2; i>0; i--, j++) {
            if(D & (1<<(p_len-1))) shift = j;
            D = (D<<1) & h_B[params.pattern[i]];
        }
    }
    double preprocess_duration = toc(&preprocess_timer_start) * 1000;
    
    cudaMemcpy(d_B, h_B, SIGMA * sizeof(int), cudaMemcpyHostToDevice);
    
        // Kernel run
    gpuErrchk( cudaEventRecord(kernel_time.start) );
    if (params.pattern_size <= 32) {
        simplified_backward_nondeterministic_dawg_unrolled<<<grid_dim, block_dim>>>(
            d_text, params.text_size, d_pattern, params.pattern_size, 
            d_B, shift, params.stride_length, d_match);
    }
    else {
        simplified_backward_nondeterministic_dawg_unrolled_large<<<grid_dim, block_dim>>>(
            d_text, params.text_size, d_pattern, params.pattern_size, 
            d_B, shift, params.stride_length, d_match);
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
