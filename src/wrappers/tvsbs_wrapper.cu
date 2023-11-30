
#include "tvsbs_wrapper.h"
#include "algos/tvsbs.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "stddef.h"
#include "util/tictoc.h"

search_info tvsbs_wrapper(search_parameters params){

        cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;

	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);

	char d_firstCh, d_lastCh;
	int **d_brBc, i;
	int **d_brBc2 = (int **)malloc(SIGMA * sizeof(int *));
	gpuErrchk( cudaMalloc((void***)&d_brBc, SIGMA * sizeof(int *)) );

	// Setup: malloc > timer start > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);

	/* Preprocessing */
	TicTocTimer preprocess_timer_start = tic();
	int h_brBc[SIGMA][SIGMA];
	
	TVSBSpreBrBc(params.pattern, params.pattern_size, h_brBc);
	d_firstCh = params.pattern[0];
	d_lastCh = params.pattern[params.pattern_size -1];
	for (i=0; i<params.pattern_size; i++) params.text[params.text_size+i]=params.text[params.text_size+params.pattern_size+i]=params.pattern[i];
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;
	
	for(i = 0; i < SIGMA; i++){
	  gpuErrchk( cudaMalloc((void**) &(d_brBc2[i]), SIGMA*sizeof(int)) ); 
	  gpuErrchk( cudaMemcpy(d_brBc2[i], h_brBc[i], SIGMA*sizeof(int), cudaMemcpyHostToDevice) );
	}
	gpuErrchk( cudaMemcpy(d_brBc, d_brBc2, SIGMA*sizeof(float *), cudaMemcpyHostToDevice) );
	
	//Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	tvsbs<<<grid_dim, block_dim>>>(
				       d_text, params.text_size, d_pattern, params.pattern_size,
				       d_brBc, d_firstCh, d_lastCh, params.stride_length, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
	// Release memory
       	gpuErrchk( cudaFree(d_brBc) );
	for(i = 0; i < SIGMA; i++){
	  gpuErrchk( cudaFree(d_brBc2[i]) ); 
	}

	return timers;
}
