
#include "br_wrapper.h"
#include "algos/br.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

search_info berry_ravindran_wrapper(search_parameters params){

	cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;
	
	//params.text[params.text_size + 1] = '\0';
	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);

	/* Preprocessing */
	TicTocTimer preprocess_timer_start = tic();
	int *brBc;
	int *d_brBc;
	int brBc_size = SIGMA * SIGMA * sizeof(int);
	brBc = (int *)malloc(brBc_size);
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;
	
	gpuErrchk( cudaMalloc(&d_brBc, brBc_size) );
	
	// Setup: malloc > timer > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);
	
	preBrBc(params.pattern, params.pattern_size, brBc);
	gpuErrchk( cudaMemcpy(d_brBc, brBc, brBc_size, 
		cudaMemcpyHostToDevice) );

	// Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	berry_ravindran<<<grid_dim, block_dim>>>(
		d_text, params.text_size, d_pattern, params.pattern_size, 
		d_brBc, params.stride_length, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
	// Release memory
	gpuErrchk( cudaFree(d_brBc) );
	free(brBc);

	return timers;
}
