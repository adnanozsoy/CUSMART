
#include "raita_wrapper.h"
#include "algos/raita.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"
#include <stdlib.h>

search_info raita_wrapper(search_parameters params){

  cuda_time kernel_time = {0}, total_time = {0};
  unsigned char *d_text, *d_pattern;
  int *d_match;
  unsigned int grid_dim, block_dim;

  setup_timers(&kernel_time, &total_time);
  get_kernel_configuration(params, &grid_dim, &block_dim);

  unsigned char *d_bmBc;
  char d_firstCh, d_middleCh, d_lastCh;
  int bmBc_malloc_size = SIGMA * sizeof(unsigned char);
  gpuErrchk( cudaMalloc(&d_bmBc, bmBc_malloc_size) );
  
  // Setup: malloc > timer start > memset/memcpy
  wrapper_setup(params, &d_text, &d_pattern, &d_match);

  /* Preprocessing */
  TicTocTimer preprocess_timer_start = tic();
  unsigned char *h_bmBc;
  h_bmBc = (unsigned char *)malloc(bmBc_malloc_size);
  
  preBmBcRAITA(params.pattern, params.pattern_size, h_bmBc);
  d_firstCh = params.pattern[0];
  d_middleCh = params.pattern[params.pattern_size/2];
  d_lastCh = params.pattern[params.pattern_size - 1];
  double preprocess_duration = toc(&preprocess_timer_start) * 1000;
  
  gpuErrchk( cudaMemcpy(d_bmBc, h_bmBc, SIGMA * sizeof(unsigned char), cudaMemcpyHostToDevice) );
  
  // Kernel run
  gpuErrchk( cudaEventRecord(kernel_time.start) );
  raita<<<grid_dim, block_dim>>>(d_text, params.text_size, d_pattern, params.pattern_size, d_bmBc,
				 d_firstCh, d_middleCh, d_lastCh, params.stride_length, d_match);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaEventRecord(kernel_time.stop) );

  search_info timers = {0};
  timers.preprocess_duration = preprocess_duration;
  // Teardown: copy match back > timer stop > free
  wrapper_teardown(params, &timers, d_text, d_pattern, d_match);

  // Release memory
  gpuErrchk( cudaFree(d_bmBc) );
  free(h_bmBc);

  return timers;
}
