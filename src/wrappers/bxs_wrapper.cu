#include "bxs_wrapper.h"
#include "algos/bxs.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

search_info bndm_extended_shift_wrapper(search_parameters params) {

    cuda_time kernel_time = {0}, total_time = {0};
    unsigned char *d_text, *d_pattern;
    int *d_match;
    unsigned int grid_dim, block_dim;

    setup_timers(&kernel_time, &total_time);
    get_kernel_configuration(params, &grid_dim, &block_dim);

    unsigned int* h_B = (unsigned int*) malloc(SIGMA * sizeof(unsigned int));
    unsigned int* d_B;
    cudaMalloc((void**)&d_B, SIGMA * sizeof(unsigned int));
    unsigned int set;
    int i;
    
    // Setup: malloc > timer start > memset/memcpy
    wrapper_setup(params, &d_text, &d_pattern, &d_match);

    /* Preprocessing */
    TicTocTimer preprocess_timer_start = tic();
    set = 1; 
    for (i=0; i<SIGMA; i++) h_B[i]=0; 
    for (i = params.pattern_size-1; i >=0; i--) { 
      h_B[params.pattern[i]] |= set; 
      set<<=1; 
      if (set==0) set=1; 
    }
    double preprocess_duration = toc(&preprocess_timer_start) * 1000;
    
    cudaMemcpy(d_B, h_B, SIGMA * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Kernel run
    gpuErrchk( cudaEventRecord(kernel_time.start) );
    bndm_extended_shift<<<grid_dim, block_dim>>>(
						d_text, params.text_size, d_pattern, params.pattern_size,
						d_B, params.stride_length, d_match);
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
