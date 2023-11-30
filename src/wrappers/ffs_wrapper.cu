
#include "ffs_wrapper.h"
#include "algos/ffs.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"


search_info forward_fast_search_wrapper(search_parameters params)
{
    cuda_time kernel_time = {0}, total_time = {0};
    unsigned char *d_text, *d_pattern;
    int *d_match;
    unsigned int grid_dim, block_dim;

    setup_timers(&kernel_time, &total_time);
    get_kernel_configuration(params, &grid_dim, &block_dim);

    int *h_bc = (int*)malloc( SIGMA * sizeof(int));
    int *h_gs = (int*)malloc( SIGMA * params.pattern_size * sizeof(int));
    unsigned char *tail = (unsigned char*)malloc((params.pattern_size + 1) * sizeof(char));
    int *d_gs, *d_bc;
    gpuErrchk( cudaMalloc(&d_bc, SIGMA * sizeof(int)) );
    gpuErrchk( cudaMalloc(&d_gs, SIGMA * params.pattern_size * sizeof(int)) );

    // Setup: malloc > timer start > memset/memcpy
    wrapper_setup(params, &d_text, &d_pattern, &d_match);

    // Preprocessing
    TicTocTimer preprocess_timer_start = tic();
    forward_suffix_function(params.pattern, params.pattern_size, h_gs, SIGMA);
    for (int i=0; i < SIGMA; i++) h_bc[i]=params.pattern_size;
    // memset(h_bc, params.pattern_size, SIGMA * sizeof(int));
    for (int j=0; j < params.pattern_size; j++)
        h_bc[params.pattern[j]]=params.pattern_size-j-1;
    for (int i = 0; i < params.pattern_size; ++i)
        tail[i] =   params.pattern[params.pattern_size-1];
    tail[params.pattern_size] = '\0';
    double preprocess_duration = toc(&preprocess_timer_start) * 1000;
    
    cudaMemcpy(d_gs, h_gs,
               SIGMA * params.pattern_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bc, h_bc, SIGMA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy((d_text + params.text_size), tail,
               (params.pattern_size+1) * sizeof(char), cudaMemcpyHostToDevice);


    // Kernel run
    gpuErrchk( cudaEventRecord(kernel_time.start) );
    forward_fast_search<<<grid_dim, block_dim>>>(
        d_text, params.text_size, d_pattern, params.pattern_size,
        d_bc, d_gs, params.stride_length, d_match);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaEventRecord(kernel_time.stop) );

    search_info timers = {0};
    timers.preprocess_duration = preprocess_duration;
    // Teardown: copy match back > timer stop > free
    wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
    // Release Memory
    cudaFree(d_bc);
    cudaFree(d_gs);
    free(h_bc);
    free(h_gs);
    free(tail);
    return timers;
}
