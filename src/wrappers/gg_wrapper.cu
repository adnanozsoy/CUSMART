#include "gg_wrapper.h"
#include "algos/gg.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

#include <stdio.h>

search_info galil_giancarlo_wrapper(search_parameters params){

	cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;

	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);

	int *h_h = (int*)malloc((params.pattern_size+1) * sizeof(int));
	int *h_next = (int*)malloc((params.pattern_size+1) * sizeof(int));
	int *h_shift = (int*)malloc((params.pattern_size+1) * sizeof(int));

	int *d_h, *d_next, *d_shift;
	gpuErrchk( cudaMalloc((void**)&d_h, (params.pattern_size+1) * sizeof(int)) );
	gpuErrchk( cudaMalloc((void**)&d_next, (params.pattern_size+1) * sizeof(int)) );
	gpuErrchk( cudaMalloc((void**)&d_shift, (params.pattern_size+1) * sizeof(int)) );

	// Setup: malloc > timer start > memset/memcpy	
	wrapper_setup(params, &d_text, &d_pattern, &d_match);
	TicTocTimer preprocess_timer_start = tic();
	int ell = 0;
	while (params.pattern[ell] == params.pattern[ell + 1] && 
	       ell < params.pattern_size) ++ell;
	
	int nd = preColussiGG(params.pattern, params.pattern_size, h_h, h_next, h_shift);
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;
	
	gpuErrchk( cudaMemcpy(d_h, h_h, 
		(params.pattern_size+1) * sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_next, h_next, 
		(params.pattern_size+1) * sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_shift, h_shift, 
		(params.pattern_size+1) * sizeof(int), cudaMemcpyHostToDevice) );

	// Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	galil_giancarlo<<<grid_dim, block_dim>>>(
		d_text, params.text_size, 
		d_pattern, params.pattern_size, 
		nd, ell, d_h, d_next, d_shift, params.stride_length, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
	// Release memory
	gpuErrchk( cudaFree(d_h) );
	gpuErrchk( cudaFree(d_next) );
	gpuErrchk( cudaFree(d_shift) );
	free(h_h);
	free(h_next);
	free(h_shift);
	return timers;
}
