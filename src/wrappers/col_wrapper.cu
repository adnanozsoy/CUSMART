#include "col_wrapper.h"
#include "algos/col.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"

#include <stdio.h>

search_info colussi_wrapper(search_parameters params){

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

	int nd = preColussi(params.pattern, params.pattern_size, h_h, h_next, h_shift);

	gpuErrchk( cudaMemcpy(d_h, h_h, 
		(params.pattern_size+1) * sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_next, h_next, 
		(params.pattern_size+1) * sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_shift, h_shift, 
		(params.pattern_size+1) * sizeof(int), cudaMemcpyHostToDevice) );

	// Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	colussi<<<grid_dim, block_dim>>>(
		d_text, params.text_size, 
		d_pattern, params.pattern_size, 
		nd, d_h, d_next, d_shift, params.stride_length, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
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
