#ifndef WRAPPER_HELPERS_H	
#define WRAPPER_HELPERS_H

#include "util/parameters.h"

typedef struct cuda_time {
	cudaEvent_t start;
	cudaEvent_t stop;
}cuda_time;

void get_kernel_configuration(
	search_parameters p, unsigned int *grid_dim, unsigned int *block_dim);

void get_kernel_configuration_shared(search_parameters p, int shared_size,
	unsigned int *grid_dim, unsigned int *block_dim);

void wrapper_setup(
	search_parameters p, unsigned char **d_text, unsigned char **d_pattern, int **d_match);

void wrapper_setup_malloc(
	search_parameters p, unsigned char **d_text, unsigned char **d_pattern, int **d_match);

void wrapper_setup_memcpy(
	search_parameters p, unsigned char **d_text, unsigned char **d_pattern, int **d_match);

void wrapper_teardown(
	search_parameters p, search_info *timers, 
	unsigned char *d_text, unsigned char *d_pattern, int *d_match);

void setup_timers(cuda_time *kernel, cuda_time *total);

#endif
