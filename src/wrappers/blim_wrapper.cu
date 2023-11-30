
#include "blim_wrapper.h"
#include "algos/blim.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"

#include <stdlib.h>
#include <string.h>

search_info bit_parallel_length_invariant_matcher_wrapper(search_parameters params)
{

    cuda_time kernel_time = {0}, total_time = {0};
    unsigned char *d_text, *d_pattern;
    int *d_match;
    unsigned int grid_dim, block_dim;

    setup_timers(&kernel_time, &total_time);
    get_kernel_configuration(params, &grid_dim, &block_dim);

    unsigned int   wsize = WORD - 1 + params.pattern_size;
    unsigned long* MM = (unsigned long*) malloc(sizeof(unsigned long)*SIGMA * wsize);
    unsigned int*  shift = (unsigned int*)malloc(sizeof(unsigned int) * SIGMA);
    unsigned int*  ScanOrder = 
    	(unsigned int*)malloc(sizeof(unsigned int) * params.pattern_size);
    unsigned int*  MScanOrder = 
    	(unsigned int*)malloc(sizeof(unsigned int) * params.pattern_size);

    unsigned long* d_MM;
    unsigned int*  d_shift;
    unsigned int*  d_ScanOrder;
    unsigned int*  d_MScanOrder;
    cudaMalloc((void**)&d_MM, sizeof(unsigned long)*SIGMA*wsize);
    cudaMalloc((void**)&d_shift, sizeof(unsigned int)*SIGMA);
    cudaMalloc((void**)&d_ScanOrder, sizeof(unsigned int)*params.pattern_size);
    cudaMalloc((void**)&d_MScanOrder, sizeof(unsigned int)*params.pattern_size);

    // Setup: malloc > timer start > memset/memcpy
    wrapper_setup(params, &d_text, &d_pattern, &d_match);

    /* Preprocessing */
    memset(MM, 0xff, sizeof(unsigned long)*SIGMA*wsize);
    for (int i=0; i<WORD; i++) {
        unsigned long tmp = 1 << i;
        for (int j=0; j<params.pattern_size; j++) {
            for (int k=0; k<SIGMA; k++) MM[((i+j)*SIGMA) + k] &= ~tmp;
            MM[ params.pattern[j] + ((i+j)*SIGMA) ] |= tmp;
        }
    }

    for (int i=0; i<SIGMA; i++) shift[i] = wsize + 1;
    for (int i=0; i<params.pattern_size; i++) shift[params.pattern[i]] = wsize - i;

    unsigned int* so  = ScanOrder;
    unsigned int* mso = MScanOrder;
    for (int i=params.pattern_size-1; i>=0; i--) {
        int k = i;
        while (k<wsize) {
            *so = k;
            *mso = SIGMA*k;
            so++;
            mso++;
            k += params.pattern_size;
        }
    }

    cudaMemcpy(d_MM, MM, sizeof(unsigned long)*SIGMA*wsize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_shift, shift, sizeof(unsigned int)*SIGMA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ScanOrder, ScanOrder,
        sizeof(unsigned int)*params.pattern_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MScanOrder, MScanOrder,
        sizeof(unsigned int)*params.pattern_size, cudaMemcpyHostToDevice);

    // Kernel run
    gpuErrchk( cudaEventRecord(kernel_time.start) );
    bit_parallel_length_invariant_matcher<<<grid_dim, block_dim>>>(
        d_text, params.text_size, d_pattern, params.pattern_size,
        d_ScanOrder, d_MScanOrder, d_MM, d_shift,
        params.stride_length, d_match);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaEventRecord(kernel_time.stop) );

    search_info timers = {0};
    // Teardown: copy match back > timer stop > free
    wrapper_teardown(params, &timers, d_text, d_pattern, d_match);

    // Free memory
    gpuErrchk( cudaFree(d_MM) );
    gpuErrchk( cudaFree(d_shift) );
    gpuErrchk( cudaFree(d_ScanOrder) );
    gpuErrchk( cudaFree(d_MScanOrder) );
    // free(MM);
    // free(shift);
    // free(ScanOrder);
    // free(MScanOrder);
    return timers;
}
