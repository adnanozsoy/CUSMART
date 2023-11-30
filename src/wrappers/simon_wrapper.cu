#include "simon_wrapper.h"
#include "algos/simon.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"

#include <stdio.h>

search_info simon_wrapper(search_parameters params){

	cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;

	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);

	List *shift_list = (List*)malloc((params.pattern_size-1) * sizeof(List));
	int ell = pre_simon(params.pattern, params.pattern_size, shift_list);
    shift_struct *h_shift = flatten_list_to_array(shift_list, params.pattern_size - 1);

    // create temporary pointers for device structure and transfer
    shift_struct *d_shift;
    int *temp_start, *temp_data;
    gpuErrchk( cudaMalloc((void**)&d_shift, sizeof(shift_struct)) );
    gpuErrchk( cudaMalloc((void**)&temp_start, 
    	(params.pattern_size - 1) * sizeof(int)) );
    gpuErrchk( cudaMemcpy(&d_shift->start, &temp_start, 
        sizeof(int*), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMalloc((void**)&temp_data, 
    	2 * (params.pattern_size - 1) * sizeof(int)) );
    gpuErrchk( cudaMemcpy(&d_shift->data, &temp_data, 
        sizeof(int*), cudaMemcpyHostToDevice) );

	// Setup: malloc > timer start > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);

    gpuErrchk( cudaMemcpy(temp_start, h_shift->start, 
        (params.pattern_size-1)*sizeof(int), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(temp_data, h_shift->data, 
        2*(params.pattern_size-1)*sizeof(int), cudaMemcpyHostToDevice) );

	// Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	simon<<<grid_dim, block_dim>>>(
		d_text, params.text_size, 
		d_pattern, params.pattern_size, 
		ell, d_shift, params.stride_length, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
    // Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
    // Release memory
	free(shift_list);
	free(h_shift->data);
	free(h_shift->start);
	free(h_shift);

	return timers;
}