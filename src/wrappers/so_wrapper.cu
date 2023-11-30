#include "so_wrapper.h"
#include "algos/so.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"
#include <stdio.h>
#include <stdlib.h>

search_info so_wrapper(search_parameters params){
  
        cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;
	
	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);
	
	unsigned int *d_S, d_lim, d_D;
	gpuErrchk( cudaMalloc(&d_S, SIGMA * sizeof(unsigned int)) );
	
	// Setup: malloc > timer start > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);
	
	/* Preprocessing */
	TicTocTimer preprocess_timer_start = tic();
	unsigned int *h_S;
	h_S = (unsigned int *)malloc(SIGMA * sizeof(unsigned int));   
	
	d_lim = preSo(params.pattern, params.pattern_size, h_S); 
	d_D = ~0;
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;
	
	gpuErrchk( cudaMemcpy(d_S, h_S, SIGMA * sizeof(unsigned int), cudaMemcpyHostToDevice) );

	// Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	if(params.pattern_size > WORD){
	      shift_or_large<<<grid_dim, block_dim>>>(d_text, params.text_size, d_pattern,
						      params.pattern_size, d_S, d_lim, d_D, params.stride_length, d_match);
	}
	else{
	      shift_or<<<grid_dim, block_dim>>>(d_text, params.text_size, d_pattern,
						params.pattern_size, d_S, d_lim, d_D, params.stride_length, d_match);
	}
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );
	
	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
	
	// Release memory
	gpuErrchk( cudaFree(d_S) );
	free(h_S); 
	
	return timers;
}
