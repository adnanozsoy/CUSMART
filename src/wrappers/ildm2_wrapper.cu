
#include "ildm2_wrapper.h"
#include "algos/ildm2.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

search_info improved_linear_dawg2_wrapper(search_parameters params)
{

    cuda_time kernel_time = {0}, total_time = {0};
    unsigned char *d_text, *d_pattern;
    int *d_match;
    unsigned int grid_dim, block_dim;

    setup_timers(&kernel_time, &total_time);
    get_kernel_configuration(params, &grid_dim, &block_dim);

    int *d_ttrans;
    int *d_ttransSMA;
    unsigned char *d_tterminal;

    gpuErrchk( cudaMalloc(&d_ttrans, 3*params.pattern_size*SIGMA*sizeof(int)) );
    gpuErrchk( cudaMalloc(&d_ttransSMA,
                          (params.pattern_size+1)*SIGMA*sizeof(int)) );
    gpuErrchk( cudaMalloc(&d_tterminal, 3*params.pattern_size*sizeof(char)) );

    // Setup: malloc > timer > memset/memcpy
    wrapper_setup(params, &d_text, &d_pattern, &d_match);

    /* Preprocessing */
    TicTocTimer preprocess_timer_start = tic();
    int *h_ttrans, *h_tlength, *h_tsuffix, *h_ttransSMA;
    unsigned char *h_tterminal;

    h_ttrans = (int *)malloc(3*params.pattern_size*SIGMA*sizeof(int));
    memset(h_ttrans, -1, 3*params.pattern_size*SIGMA*sizeof(int));
    h_tlength = (int *)calloc(3*params.pattern_size, sizeof(int));
    h_tsuffix = (int *)calloc(3*params.pattern_size, sizeof(int));
    h_tterminal = (unsigned char *)calloc(3*params.pattern_size, sizeof(char));

    unsigned char *xR = (unsigned char*) malloc (sizeof(char)*(params.pattern_size+1));
    for (int i=0; i<params.pattern_size;
         i++) xR[i] = params.pattern[params.pattern_size-i-1];
    xR[params.pattern_size] = '\0';

    buildSimpleSuffixAutomaton(xR, params.pattern_size, h_ttrans, h_tlength,
                               h_tsuffix, h_tterminal);

    h_ttransSMA = (int *)malloc((params.pattern_size+1)*SIGMA*sizeof(int));
    memset(h_ttransSMA, -1, (params.pattern_size+1)*SIGMA*sizeof(int));
    preSMA(params.pattern, params.pattern_size, h_ttransSMA);
    double preprocess_duration = toc(&preprocess_timer_start) * 1000;
    
    gpuErrchk( cudaMemcpy(d_ttrans, h_ttrans,
                          3*params.pattern_size*SIGMA*sizeof(int), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_ttransSMA, h_ttransSMA,
                          (params.pattern_size+1)*SIGMA*sizeof(int), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_tterminal, h_tterminal,
                          3*params.pattern_size*sizeof(char), cudaMemcpyHostToDevice) );

    // Kernel run
    gpuErrchk( cudaEventRecord(kernel_time.start) );
    improved_linear_dawg2<<<grid_dim, block_dim>>>(
        d_text, params.text_size, d_pattern, params.pattern_size,
        d_ttrans, d_ttransSMA, d_tterminal, params.stride_length, d_match);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaEventRecord(kernel_time.stop) );

    search_info timers = {0};
    timers.preprocess_duration = preprocess_duration;
    // Teardown: copy match back > timer stop > free
    wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
    // Release memory
    gpuErrchk( cudaFree(d_ttrans) );
    gpuErrchk( cudaFree(d_ttransSMA) );
    gpuErrchk( cudaFree(d_tterminal) );
    free(h_ttrans);
    free(h_tlength);
    free(h_tsuffix);
    free(h_tterminal);
    free(h_ttransSMA);

    return timers;
}
