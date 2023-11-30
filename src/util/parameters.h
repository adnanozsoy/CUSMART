#ifndef PARAMETERS_H
#define PARAMETERS_H

#define GPU_TEST 1<<0
#define CPU_TEST 1<<1

typedef struct search_info
{
	float total_duration;
	float kernel_duration;
	float reduce_duration;
	float preprocess_duration;
	float setupcopy_duration;
} search_info;

typedef struct search_parameters
{
	unsigned char *text;
	unsigned long int text_size;
	unsigned char *pattern;
	int pattern_size;
	int *match;
	int stride_length;
	int pinned_memory;
	int constant_memory;
	int test_flags; // 01 - gpu, 10 - cpu 
	int gpu_reduction;
	int block_dim;
	int search_average_runs;
} search_parameters;

#endif