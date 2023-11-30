
#include "nsn_wrapper.h"
#include "algos/nsn.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

#include <stdlib.h>

search_info nsn_wrapper(search_parameters params){
  
        cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;
	
	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);
	
	unsigned char d_k, d_ell;
	
	// Setup: malloc > timer start > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);
	
	/* Preprocessing */
	TicTocTimer preprocess_timer_start = tic();
	if (params.pattern[0] == params.pattern[1]) {
	      d_k = 2;
	      d_ell = 1;
	}
	else {
	      d_k = 1;
	      d_ell = 2;
	}
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;
	
	// Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	not_so_naive<<<grid_dim, block_dim>>>(d_text, params.text_size, d_pattern,
				       params.pattern_size, d_k, d_ell, params.stride_length, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );
	
	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
	
	// Release memory
	
	return timers;
}
