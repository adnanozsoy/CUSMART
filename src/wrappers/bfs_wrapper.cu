
#include "bfs_wrapper.h"
#include "algos/bfs.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

search_info backward_fast_search_wrapper(search_parameters params)
{
	cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;

	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);

	int *h_gs = (int*) malloc(SIGMA * (params.pattern_size+1) * sizeof(int));
	int *h_bc = (int*) malloc(SIGMA * sizeof(int));
	int *d_gs, *d_bc;
	cudaMalloc((void**)&d_gs, SIGMA * (params.pattern_size+1) * sizeof(int));
	cudaMalloc((void**)&d_bc, SIGMA * sizeof(int));
	char* tail = (char*) malloc((params.pattern_size + 1) * sizeof(char));

	// Setup: malloc > timer start > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);

	/* Preprocessing */
	TicTocTimer preprocess_timer_start = tic();
	for (int i = 0; i < SIGMA; ++i)
		h_bc[i] = params.pattern_size;
   	for (int i = 0; i < params.pattern_size; ++i) 
   		h_bc[params.pattern[i]] = params.pattern_size - i - 1;
   	// for(int i=0;i<params.pattern_size;i++)
   	// 	params.text[params.text_size+i]=params.pattern[i];

   	PreBFS(params.pattern, params.pattern_size, h_gs);
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;
	
   	cudaMemcpy(d_gs, h_gs, 
   		SIGMA * (params.pattern_size+1) * sizeof(int), cudaMemcpyHostToDevice);
   	cudaMemcpy(d_bc, h_bc, SIGMA * sizeof(int), cudaMemcpyHostToDevice);
   	cudaMemcpy((d_text + params.text_size), params.pattern, 
   		params.pattern_size * sizeof(char), cudaMemcpyHostToDevice);

	//Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	backward_fast_search<<<grid_dim, block_dim>>>(
		d_text, params.text_size, d_pattern, params.pattern_size, 
		d_bc, d_gs, params.stride_length, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
	// Free Memory
	cudaFree(d_gs);
	cudaFree(d_bc);
	free(h_gs);
	free(h_bc);
	free(tail);
	return timers;
}
