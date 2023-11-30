
#include "qs_wrapper.h"
#include "algos/qs.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"


search_info quicksearch_wrapper(search_parameters params){
	
	cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;
	
	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);

	int *d_bmBc;
	gpuErrchk( cudaMalloc(&d_bmBc, SIGMA * sizeof(int)) );

	// Setup: malloc > timer start > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);
	

	/* Preprocessing */
	TicTocTimer preprocess_timer_start = tic();
	int h_bmBc[SIGMA]; 
	preQsBc(params.pattern, params.pattern_size, h_bmBc);
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;
	
	gpuErrchk( cudaMemcpy(d_bmBc, h_bmBc, SIGMA * sizeof(int), cudaMemcpyHostToDevice) );

	//Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	quicksearch<<<grid_dim, block_dim>>>(
		d_text, params.text_size, d_pattern, params.pattern_size, 
		d_bmBc, params.stride_length, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
	
	// Release memory
	gpuErrchk( cudaFree(d_bmBc) );

	return timers;
}
