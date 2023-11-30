
#include "fs_wrapper.h"
#include "algos/fs.cuh"
#include "wrapper_helpers.h"
#include "util/tictoc.h"
#include "util/cutil.cuh"

search_info fast_search_wrapper(search_parameters params){
	
	cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;
	
	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);

	int *d_gs, *d_bc;
	gpuErrchk( cudaMalloc(&d_gs, (params.pattern_size + 1) * sizeof(int)) );
	gpuErrchk( cudaMalloc(&d_bc, SIGMA * sizeof(int)) );

	// Setup: malloc > timer start > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);
	

	/* Preprocessing */
	TicTocTimer preprocess_timer_start = tic();
	//gs does not need to be allocated as XSIZE according to Pre_GS it 
	//should be allocated as m+1
	int bc[SIGMA]; 
	int *gs = (int*)malloc((params.pattern_size + 1) * sizeof(int));
	
	for (int a=0; a < SIGMA; a++) bc[a] = params.pattern_size;
	for (int j=0; j < params.pattern_size; j++) bc[params.pattern[j]] = params.pattern_size - j - 1;
	Pre_GS(params.pattern, params.pattern_size, gs);
	//char ch = params.pattern[params.pattern_size - 1];
	//for (int i=0; i < params.pattern_size; i++) params.text[params.text_size + i] = ch;
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;
	
	gpuErrchk( cudaMemcpy(d_gs, gs, (params.pattern_size + 1) * sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_bc, bc, SIGMA * sizeof(int), cudaMemcpyHostToDevice) );

	//Kernel run
	//if ( !memcmp(x,y,m) ) count++; 
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	fast_search<<<grid_dim, block_dim>>>(
		d_text, params.text_size, d_pattern, params.pattern_size, 
		d_bc, d_gs, params.stride_length, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
	// Release memory
	gpuErrchk( cudaFree(d_bc) );
	gpuErrchk( cudaFree(d_gs) );
	free(gs);

	return timers;
}
