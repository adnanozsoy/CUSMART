
#include "tw_wrapper.h"
#include "algos/tw.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"
#include <string.h>

search_info two_way_wrapper(search_parameters params){
	
	cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;
	
	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);

	// Setup: malloc > timer start > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);
	

	/* Preprocessing */
	TicTocTimer preprocess_timer_start = tic();
	int i, j, per, ell, p, q;
	i = maxSuf(params.pattern, params.pattern_size, &p);
	j = maxSufTilde(params.pattern, params.pattern_size, &q);
	if (i > j) {
	  ell = i;
	  per = p;
	}
	else {
	  ell = j;
	  per = q;
	}
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;
	
	//Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	
	if (memcmp(params.pattern, params.pattern + per, ell + 1) == 0) 
		two_way1<<<grid_dim, block_dim>>>(
			d_text, params.text_size, d_pattern, params.pattern_size, 
			per, ell, params.stride_length, d_match);
	else
		two_way2<<<grid_dim, block_dim>>>(
			d_text, params.text_size, d_pattern, params.pattern_size, 
			per, ell, params.stride_length, d_match);
		
		
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);

	return timers;
}
