
#include "dfa_wrapper.h"
#include "algos/dfa.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

#include <stdlib.h>
#include <stdio.h>


search_info deterministic_finite_automaton_wrapper(search_parameters params){

	cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;
	
	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);

	/* Preprocessing */
	TicTocTimer preprocess_timer_start = tic();
	int *ttransSMA;
	long transSMA_size = (params.pattern_size + 1)*SIGMA*sizeof(int);
	ttransSMA = (int *)malloc(transSMA_size);
	int *d_ttransSMA;
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;
	
	gpuErrchk( cudaMalloc(&d_ttransSMA, transSMA_size) );
	
	// Setup: malloc > timer > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);

	memset(ttransSMA, -1, transSMA_size);
	preSMA(params.pattern, params.pattern_size, ttransSMA);
	gpuErrchk( cudaMemcpy(d_ttransSMA, ttransSMA, 
		transSMA_size, cudaMemcpyHostToDevice) );

	// Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	deterministic_finite_automaton<<<grid_dim, block_dim>>>(
		d_text, params.text_size, d_pattern, params.pattern_size, 
		d_ttransSMA, params.stride_length, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
	// Release memory
	gpuErrchk( cudaFree(d_ttransSMA) );
	free(ttransSMA);

	return timers;
}
