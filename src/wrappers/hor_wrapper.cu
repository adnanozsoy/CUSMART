
#include "hor_wrapper.h"
#include "algos/hor.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

#include <stdlib.h>

search_info horspool_wrapper(search_parameters params){

        cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
  	int *d_match;
	unsigned int grid_dim, block_dim;

	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);

	unsigned char *d_hbc;
	int hbc_size = SIGMA * sizeof(unsigned char);
	gpuErrchk( cudaMalloc((void**)&d_hbc, hbc_size) );  
	
        // Setup: malloc > timer start > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);   
	
	/* Preprocessing */
	TicTocTimer preprocess_timer_start = tic();
	unsigned char *hbc;
	hbc = (unsigned char *)malloc(hbc_size);
	
      	pre_horspool(params.pattern, params.pattern_size, hbc);
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;
	
	gpuErrchk( cudaMemcpy(d_hbc, hbc, hbc_size, cudaMemcpyHostToDevice) );

	// Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	horspool<<<grid_dim, block_dim>>>(d_text, params.text_size, d_pattern,
		      params.pattern_size, d_hbc, params.stride_length, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);

	// Release memory
	gpuErrchk( cudaFree(d_hbc) );
	free(hbc);

	return timers;
}
