
#include "ms_wrapper.h"
#include "algos/ms.cuh"
#include "wrapper_helpers.h"
#include "util/cutil.cuh"
#include "util/tictoc.h"

search_info maximal_shift_wrapper(search_parameters params){

  cuda_time kernel_time = {0}, total_time = {0};
  unsigned char *d_text, *d_pattern;
  int *d_match;
  unsigned int grid_dim, block_dim;

  setup_timers(&kernel_time, &total_time);
  get_kernel_configuration(params, &grid_dim, &block_dim);

  int *d_qsBc, *d_adaptedGs;
  patternS *d_pat;
  gpuErrchk( cudaMalloc(&d_qsBc, SIGMA * sizeof(int)) );
  gpuErrchk( cudaMalloc(&d_adaptedGs, (params.pattern_size+1) * sizeof(int)) );
  gpuErrchk( cudaMalloc(&d_pat, (params.pattern_size+1) * sizeof(patternS)) );

  // Setup: malloc > timer start > memset/memcpy
  wrapper_setup(params, &d_text, &d_pattern, &d_match);

  /* Preprocessing */
  TicTocTimer preprocess_timer_start = tic();
  int *h_adaptedGs = (int *)malloc((params.pattern_size+1) * sizeof(int));
  patternS *h_pat = (patternS *)malloc((params.pattern_size+1) * sizeof(patternS));
  int h_qsBc[SIGMA];
  
  computeMinShift(params.pattern ,params.pattern_size);
  orderPatternMS(params.pattern ,params.pattern_size, maxShiftPcmp, h_pat);
  preQsBcMS(params.pattern ,params.pattern_size, h_qsBc);
  preAdaptedGsMS(params.pattern ,params.pattern_size, h_adaptedGs, h_pat);
  double preprocess_duration = toc(&preprocess_timer_start) * 1000;
  
  gpuErrchk( cudaMemcpy(d_qsBc, h_qsBc, SIGMA * sizeof(int), cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(d_adaptedGs, h_adaptedGs, (params.pattern_size + 1) * sizeof(int), cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(d_pat, h_pat, (params.pattern_size + 1) * sizeof(patternS), cudaMemcpyHostToDevice) );
  
  //Kernel run
  gpuErrchk( cudaEventRecord(kernel_time.start) );
  maximal_shift<<<grid_dim, block_dim>>>(
				       d_text, params.text_size, d_pattern, params.pattern_size,
				       d_qsBc, d_adaptedGs, d_pat, params.stride_length, d_match);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaEventRecord(kernel_time.stop) );

  search_info timers = {0};
  timers.preprocess_duration = preprocess_duration;
  // Teardown: copy match back > timer stop > free
  wrapper_teardown(params, &timers, d_text, d_pattern, d_match);

  // Release memory
  gpuErrchk( cudaFree(d_qsBc) );
  gpuErrchk( cudaFree(d_adaptedGs) );
  gpuErrchk( cudaFree(d_pat) );
  free(h_adaptedGs);
  free(h_pat);
  return timers;
}
