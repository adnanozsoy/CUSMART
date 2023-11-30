#include "rcolussi_wrapper.h"
#include "algos/rcolussi.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"
#include <stdio.h>

search_info reverse_colussi_wrapper(search_parameters params){

	cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;

	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);

	int *h_h = (int*)malloc((params.pattern_size+1) * sizeof(int));
	int *h_rcBc = (int*)malloc(SIGMA * (params.pattern_size+1) * sizeof(int));
	int *h_rcGs = (int*)malloc((params.pattern_size+1) * sizeof(int));

	int *d_h, *d_rcBc, *d_rcGs;
	gpuErrchk( cudaMalloc((void**)&d_h, (params.pattern_size+1) * sizeof(int)) );
	gpuErrchk( cudaMalloc((void**)&d_rcBc, SIGMA * (params.pattern_size+1) * sizeof(int)) );
	gpuErrchk( cudaMalloc((void**)&d_rcGs, (params.pattern_size+1) * sizeof(int)) );

	// Setup: malloc > timer start > memset/memcpy	
	wrapper_setup(params, &d_text, &d_pattern, &d_match);

	TicTocTimer preprocess_timer_start = tic();
	preRc(params.pattern, params.pattern_size, h_h, h_rcBc, h_rcGs);
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;
	
	gpuErrchk( cudaMemcpy(d_h, h_h, 
		(params.pattern_size+1) * sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_rcBc, h_rcBc, 
		SIGMA * (params.pattern_size+1) * sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_rcGs, h_rcGs, 
		(params.pattern_size+1) * sizeof(int), cudaMemcpyHostToDevice) );

	// Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	reverse_colussi<<<grid_dim, block_dim>>>(
		d_text, params.text_size, 
		d_pattern, params.pattern_size, 
		d_h, d_rcBc, d_rcGs, params.stride_length, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
	// Release memory
	gpuErrchk( cudaFree(d_h) );
	gpuErrchk( cudaFree(d_rcBc) );
	gpuErrchk( cudaFree(d_rcGs) );
	free(h_h);
	free(h_rcBc);
	free(h_rcGs);
	return timers;
}
