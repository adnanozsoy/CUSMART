
#include "dfah_wrapper.h"
#include "algos/dfah.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

#include <stdio.h>
#include <stdlib.h>


search_info high_deterministic_finite_automaton_wrapper(search_parameters params){

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

	int threadsPerBlock = 1024;
	unsigned long blocksPerGrid = divUp(params.text_size, 1024);

	memset(ttransSMA, -1, transSMA_size);
	preHSMA(params.pattern, params.pattern_size, ttransSMA);
	gpuErrchk(cudaMemcpy(d_ttransSMA, ttransSMA, 
		transSMA_size, cudaMemcpyHostToDevice));

	// Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	high_deterministic_finite_automaton<<<blocksPerGrid, threadsPerBlock>>>(
		d_text, params.text_size, d_pattern, params.pattern_size, 
		d_ttransSMA, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
	// Release memory
	gpuErrchk( cudaFree(d_ttransSMA) );
	return timers;
}
