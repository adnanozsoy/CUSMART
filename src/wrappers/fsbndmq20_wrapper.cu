
#include "fsbndmq20_wrapper.h"
#include "algos/fsbndmq20.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

search_info forward_simplified_bndm_qgram_schar_wrapper(
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
    unsigned int *h_B = (unsigned int*)malloc(SIGMA * sizeof(unsigned int));

    // Setup: malloc > timer start > memset/memcpy
    wrapper_setup(params, &d_text, &d_pattern, &d_match);

    /* Preprocessing */
    TicTocTimer preprocess_timer_start = tic();
    int Q = 2;
    int F = 0;
    int set = 0;
    int plen = params.pattern_size+F>WORD ?  WORD-F : params.pattern_size;
    for (int j=0; j<F; j++) set = (set << 1) | 1;
    for (int i=0; i<SIGMA; i++) h_B[i]=set;
    for (int i = 0; i < plen; ++i)
        h_B[params.pattern[i]] |= (1<<(plen-i-1+F));
    int mm = plen-Q+F;
    int sh = plen-Q+F+1;
    int m1 = plen-1;

    double preprocess_duration = toc(&preprocess_timer_start) * 1000;

    cudaMemcpy(d_B, h_B, SIGMA * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel run
    gpuErrchk( cudaEventRecord(kernel_time.start) );

    forward_simplified_bndm_qgram_schar<<<grid_dim, block_dim>>>(
        d_text, params.text_size, d_pattern, params.pattern_size,
        d_B, mm, sh, m1, params.stride_length, d_match);

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
