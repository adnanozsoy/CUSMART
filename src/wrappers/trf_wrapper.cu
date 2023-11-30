
#include "trf_wrapper.h"
#include "algos/trf.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

search_info turbo_reverse_factor_wrapper(search_parameters params){

	cuda_time kernel_time = {0}, total_time = {0};
	unsigned char *d_text, *d_pattern;
	int *d_match;
	unsigned int grid_dim, block_dim;
	
	setup_timers(&kernel_time, &total_time);
	get_kernel_configuration(params, &grid_dim, &block_dim);
	
        int *d_ttrans, *d_tshift, *d_mpNext;
        unsigned char *d_tterminal;
	int size = 2 * params.pattern_size + 3;
        int ttrans_size = size*SIGMA*sizeof(int);
	gpuErrchk( cudaMalloc(&d_ttrans, ttrans_size) );
	gpuErrchk( cudaMalloc(&d_tterminal, size*sizeof(char)) );	
	gpuErrchk( cudaMalloc(&d_tshift, size*SIGMA*sizeof(int)) );
	gpuErrchk( cudaMalloc(&d_mpNext, (params.pattern_size+1)*sizeof(int)) );
	
	// Setup: malloc > timer > memset/memcpy
	wrapper_setup(params, &d_text, &d_pattern, &d_match);

	/* Preprocessing */
	TicTocTimer preprocess_timer_start = tic();
	int  *ttrans, *tlength, *tposition, *tsuffix, *tshift, *mpNext;
	unsigned char *tterminal;
	
	mpNext = (int *)malloc((params.pattern_size+1)*sizeof(int));
	ttrans = (int *)malloc(ttrans_size);
	tshift = (int *)malloc(size*SIGMA*sizeof(int));
	tlength = (int *)calloc(size, sizeof(int));
	tposition = (int *)calloc(size, sizeof(int));
	tsuffix = (int *)calloc(size, sizeof(int));
	tterminal = (unsigned char *)calloc(size, sizeof(char));
	memset(ttrans, -1, ttrans_size);
	buildSuffixAutomaton4TRF(params.pattern, params.pattern_size, ttrans, tlength, tposition, tsuffix, tterminal, tshift);
	preMpforTRF(params.pattern, params.pattern_size, mpNext);
	double preprocess_duration = toc(&preprocess_timer_start) * 1000;
	
	gpuErrchk( cudaMemcpy(d_ttrans, ttrans, 
			      ttrans_size, cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_tterminal, tterminal, 
			      size*sizeof(char), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_tshift, tshift,
			      size*SIGMA*sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_mpNext, mpNext,
			      (params.pattern_size+1)*sizeof(int), cudaMemcpyHostToDevice) );
	// Kernel run
	gpuErrchk( cudaEventRecord(kernel_time.start) );
	turbo_reverse_factor<<<grid_dim, block_dim>>>(
						d_text, params.text_size, d_pattern, params.pattern_size, 
						d_ttrans, d_tterminal, d_tshift, d_mpNext, params.stride_length, d_match);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaEventRecord(kernel_time.stop) );

	search_info timers = {0};
	timers.preprocess_duration = preprocess_duration;
	// Teardown: copy match back > timer stop > free
	wrapper_teardown(params, &timers, d_text, d_pattern, d_match);
	// Release memory
	gpuErrchk( cudaFree(d_ttrans) );
	gpuErrchk( cudaFree(d_tterminal) );
	gpuErrchk( cudaFree(d_tshift) );
	gpuErrchk( cudaFree(d_mpNext) );
	free(mpNext);
	free(tshift);
	free(ttrans);
	free(tlength);
	free(tsuffix);
	free(tterminal);
	
	return timers;
}
