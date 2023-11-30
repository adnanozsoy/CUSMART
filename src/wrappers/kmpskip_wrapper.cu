#include "kmpskip_wrapper.h"
#include "algos/kmpskip.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

search_info kmpskip_wrapper(search_parameters params){

        cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;
	
	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);
	
	int *d_kmpNext, *d_list, *d_mpNext, *d_z;
	gpuErrchk( cudaMalloc(&d_kmpNext, (params.pattern_size+1) * sizeof(int)) );
	gpuErrchk( cudaMalloc(&d_list, (params.pattern_size+1) * sizeof(int)) );
	gpuErrchk( cudaMalloc(&d_mpNext, (params.pattern_size+1) * sizeof(int)) );
	gpuErrchk( cudaMalloc(&d_z, SIGMA * sizeof(int)) );
	
	// Setup: malloc > timer start > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);
	
	/* Preprocessing */
	TicTocTimer preprocess_timer_start = tic();
	int * h_kmpNext = (int *)malloc((params.pattern_size+1) * sizeof(int));
	int *h_list = (int *)malloc((params.pattern_size+1) * sizeof(int));
	int *h_mpNext = (int *)malloc((params.pattern_size+1) * sizeof(int)); 
	int h_z[SIGMA], i;
	
	preMp(params.pattern, params.pattern_size, h_mpNext);
	preKmp(params.pattern, params.pattern_size, h_kmpNext);
	memset(h_z, -1, SIGMA*sizeof(int));
	memset(h_list, -1, params.pattern_size*sizeof(int));
	h_z[params.pattern[0]] = 0;
	for (i = 1; i < params.pattern_size; ++i) {
	     h_list[i] = h_z[params.pattern[i]];
	     h_z[params.pattern[i]] = i;
	}
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;
	
	gpuErrchk( cudaMemcpy(d_kmpNext, h_kmpNext, (params.pattern_size+1) * sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_list, h_list, (params.pattern_size+1) * sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_mpNext, h_mpNext, (params.pattern_size+1) * sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_z, h_z, SIGMA * sizeof(int), cudaMemcpyHostToDevice) );
	
	//Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	kmpskip<<<grid_dim, block_dim>>>(
					 d_text, params.text_size, d_pattern, params.pattern_size,
					 d_kmpNext, d_list, d_mpNext, d_z, params.stride_length, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );
	
	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
	
	// Release memory
	gpuErrchk( cudaFree(d_kmpNext) );
	gpuErrchk( cudaFree(d_list) );
	gpuErrchk( cudaFree(d_mpNext) );
	gpuErrchk( cudaFree(d_z) );
	free(h_kmpNext);
	free(h_list);
	free(h_mpNext);
	
	return timers;
}
