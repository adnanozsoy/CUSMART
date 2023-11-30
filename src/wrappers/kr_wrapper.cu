
#include "kr_wrapper.h"
#include "algos/kr.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

search_info karp_rabin_wrapper(search_parameters params){
	
	cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;
       
	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);
	
	/* Preprocessing */
	TicTocTimer preprocess_timer_start = tic();
	int hash_factor, hpattern, i; 
	for (hash_factor = i = 1; i < params.pattern_size; ++i)
		hash_factor = (hash_factor<<1);

	for (hpattern = i = 0; i < params.pattern_size; ++i) {
		hpattern = ((hpattern<<1) + params.pattern[i]);
	}	
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;

	// Setup: malloc > timer start > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);
	
	//Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	karp_rabin<<<grid_dim, block_dim>>>(
		d_text, params.text_size, d_pattern, params.pattern_size, 
		hash_factor, hpattern, params.stride_length, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);

	return timers;
}
