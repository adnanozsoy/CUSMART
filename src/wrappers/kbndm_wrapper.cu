#include "kbndm_wrapper.h"
#include "algos/kbndm.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"
#include "stddef.h"

search_info factorized_backward_nondeterministic_dawg_matching_wrapper(search_parameters params){

        cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;

	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);
	
        unsigned int d_M;
        unsigned int *d_L, **d_B;
        unsigned int **d_B2 = (unsigned int **)malloc(SIGMA * sizeof(unsigned int *));
        gpuErrchk( cudaMalloc((void***)&d_B, SIGMA * sizeof(unsigned int *)) );
        gpuErrchk( cudaMalloc((void**)&d_L, SIGMA * sizeof(unsigned int)) );
	
	// Setup: malloc > timer start > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);

	/* Preprocessing */
	TicTocTimer preprocess_timer_start = tic();
        int i, k, m1;
        int beg, end;
        unsigned int h_B[SIGMA][SIGMA] = {{0}};
        unsigned int L[SIGMA] = {0};
	/* Preprocessing */
	end = 0;
	for (k = 0; k < (sizeof(unsigned int)*8)-1; k++) {
	  char occ[SIGMA] = {0};
	  while (end < params.pattern_size && occ[params.pattern[end]] == 0) {
	    occ[params.pattern[end]] = 1;
	    end++;
	  }
	}
	m1 = end;
	end = k = 0;
	beg = 0;
	for (;;) {
	  char occ[SIGMA] = {0};
	  while (end < m1 && occ[params.pattern[m1-1-end]] == 0) {
	    occ[params.pattern[m1-1-end]] = 1;
	    end++;
	  }
	  for (i = beg+1; i < end; i++)
	    h_B[params.pattern[m1-i]][params.pattern[m1-1-i]] |= 1L << k;
	  if (end < m1) {
	    h_B[params.pattern[m1-end]][params.pattern[m1-1-end]] |= 1L << k;
	    L[params.pattern[m1-end]] |= 1L << k;
	  } else {
	    d_M = 1L << k;
	    if (end > beg+1) {
	      L[params.pattern[m1-end+1]] |= 1L << k;
	      d_M <<= 1;
	    }
	    break;
	  }
	  beg = end;
	  k++;
	}
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;
        for(i = 0; i < SIGMA; i++){
          gpuErrchk( cudaMalloc((void**) &(d_B2[i]), SIGMA*sizeof(unsigned int)) );
          gpuErrchk( cudaMemcpy(d_B2[i], h_B[i], SIGMA*sizeof(unsigned int), cudaMemcpyHostToDevice) );
        }

        cudaMemcpy(d_L, L, SIGMA * sizeof(unsigned int), cudaMemcpyHostToDevice);
        gpuErrchk( cudaMemcpy(d_B, d_B2, SIGMA*sizeof(float *), cudaMemcpyHostToDevice) );
	
	//Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
        factorized_backward_nondeterministic_dawg<<<grid_dim, block_dim>>>(
                                      d_text, params.text_size, d_pattern, params.pattern_size,
                                      m1, d_M, d_B, d_L, params.stride_length, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );
	
	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
	// Release memory
        gpuErrchk( cudaFree(d_L) );
        gpuErrchk( cudaFree(d_B) );
        for(i = 0; i < SIGMA; i++){
          gpuErrchk( cudaFree(d_B2[i]) );
        }
	
	return timers;
	
}
