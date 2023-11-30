
#include "smoa_wrapper.h"
#include "algos/smoa.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"


search_info string_matching_ordered_alphabet_wrapper(search_parameters params){
	
	cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;
	
	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);

	// Setup: malloc > timer start > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);
	

	//Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	
 
	string_matching_ordered_alphabet<<<grid_dim, block_dim>>>(
			d_text, params.text_size, d_pattern, params.pattern_size, 
			params.stride_length, d_match);		
		
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);

	return timers;
}
