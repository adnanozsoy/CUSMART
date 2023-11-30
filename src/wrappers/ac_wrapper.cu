#include "ac_wrapper.h"
#include "algos/ac.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

#include <stdio.h>

search_info apostolico_crochemore_wrapper(search_parameters params)
{
	cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;
	
	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);
	
	int *d_shift_array;
	int *shift_array = (int*)malloc((params.pattern_size+1)*sizeof(int));
	gpuErrchk( cudaMalloc((void**)&d_shift_array,
			      (params.pattern_size+1) * sizeof(int)) );
	
	// Setup: malloc > timer start > memset/memcpy	
	wrapper_setup(params, &d_text, &d_pattern, &d_match);

	/* Preprocessing */
	TicTocTimer preprocess_timer_start = tic();
	preKmpAC(params.pattern, params.pattern_size, shift_array);
	int ell;
	for (ell = 1; ell < params.pattern_size && params.pattern[ell - 1] == params.pattern[ell]; ell++);
    if (ell == params.pattern_size)
        ell = 0;
    double preprocess_duration = toc(&preprocess_timer_start) * 1000;

	gpuErrchk( cudaMemcpy(d_shift_array, shift_array, 
			      (params.pattern_size+1) * sizeof(int), cudaMemcpyHostToDevice) );
	
	// Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	apostolico_crochemore<<<grid_dim, block_dim>>>(
		d_text, params.text_size, 
		d_pattern, params.pattern_size, 
		ell, d_shift_array, params.stride_length, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
	// Release memory
	gpuErrchk( cudaFree(d_shift_array) );
	free(shift_array);
	return timers;
}
