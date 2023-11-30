
#include "ssabs_wrapper.h"
#include "algos/ssabs.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"
#include <stdlib.h>

search_info ssabs_wrapper(search_parameters params){
	
	cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;
	
	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);

	int *d_qsBc;
	gpuErrchk( cudaMalloc(&d_qsBc, SIGMA * sizeof(int)) );

	// Setup: malloc > timer start > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);
	
	/* Preprocessing */
	TicTocTimer preprocess_timer_start = tic();
	int *h_qsBc = (int*)malloc(SIGMA * sizeof(int));
	unsigned char firstCh, lastCh;
	int i;
	
	preQsBcSSABS(params.pattern, params.pattern_size, h_qsBc);
	firstCh = params.pattern[0]; 
	lastCh = params.pattern[params.pattern_size -1]; 
	for (i=0; i<params.pattern_size; i++) params.text[params.text_size+i]=lastCh;
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;
	
	gpuErrchk( cudaMemcpy(d_qsBc, h_qsBc, SIGMA * sizeof(int), cudaMemcpyHostToDevice) );

	//Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	ssabs<<<grid_dim, block_dim>>>(
		d_text, params.text_size, d_pattern, params.pattern_size, 
		d_qsBc, firstCh, lastCh, params.stride_length, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
	// Release memory
	gpuErrchk( cudaFree(d_qsBc) );
	free(h_qsBc);

	return timers;
}
