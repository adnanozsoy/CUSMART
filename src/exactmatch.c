
#include "exactmatch.h"

#include "util/tictoc.h"
#include "util/device.h"
#include "wrappers/wrapper_list.h"
#include "srlalgos/algorithm_list.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


void parse_tagstring(char*** alg_list, int* alg_count, const char* tagstr){
    size_t slen = strlen(tagstr);
    int algcount = 1;
    for(int i = 0; i < slen; i++) if(tagstr[i]==',') algcount++;
    *alg_count = algcount;
	*alg_list = (char**)malloc(algcount * sizeof(char*));
  	for(int i=0, cursor=0, argnum=0; i<slen; i++){
    	if(tagstr[i]==',' || i == slen-1){
      		if(i == slen-1) i++;
      		(*alg_list)[argnum] = (char*)malloc((i-cursor+1)*sizeof(char));
      		memcpy((*alg_list)[argnum], tagstr+cursor, (i-cursor)*sizeof(char));
			(*alg_list)[argnum][i - cursor] = '\0';
      		cursor = i+1;
      		argnum++;
    	}
	}
}

static int gpu_test(char* tag, struct search_parameters params){
	int alg_found = 0;
	int count;
	float reduce_temp;
	struct wrapper_info wrapper;
	for (int i = 0; i < wrapper_list_len; i++)
	{
		wrapper = wrapper_list[i];
		if (!strcmp(wrapper.tag, tag) || !strcmp(tag, "all"))
		{
			search_info timers_avg = {0};
			for (int i = 0; i < params.search_average_runs; ++i)
			{
				device_reset();
				// Call search function
				search_info timers = wrapper.search(params);
	
				count = 0;
				if (params.gpu_reduction) {
					count = params.match[0];
				}
				else {
					tic();
					for (unsigned long i = 0; i < params.text_size; i++)
						count += params.match[i];		
					reduce_temp = toc(0) * 1000;
					timers_avg.reduce_duration += reduce_temp;
					timers_avg.total_duration += reduce_temp;
				}
				timers_avg.kernel_duration += timers.kernel_duration;
				timers_avg.reduce_duration += timers.reduce_duration;
				timers_avg.total_duration  += timers.total_duration;
				timers_avg.setupcopy_duration  += timers.setupcopy_duration;
				timers_avg.preprocess_duration  += timers.preprocess_duration;
			}
			timers_avg.kernel_duration /= params.search_average_runs;
			timers_avg.reduce_duration /= params.search_average_runs;
			timers_avg.total_duration  /= params.search_average_runs;
			timers_avg.setupcopy_duration  /= params.search_average_runs;
			timers_avg.preprocess_duration  /= params.search_average_runs;
			printf(" %-10s | %-45s | %10d | %10.4f | %10.4f | %10.4f | %10.4f | %10.4f \n", 
				wrapper.tag, wrapper.name, count, timers_avg.kernel_duration, 
				timers_avg.setupcopy_duration, timers_avg.preprocess_duration, 
				timers_avg.reduce_duration, timers_avg.total_duration);
			alg_found = 1;
		}
	}
	return alg_found;
}

static int cpu_test(char* tag, struct search_parameters params){
	double func_duration, reduce_duration;
	int alg_found = 0;
	int count;
	struct algorithm_info algorithm;
	for (int i = 0; i < algorithm_list_len; i++)
	{
		algorithm = algorithm_list[i];
		if (!strcmp(algorithm.tag, tag) || !strcmp(tag, "all"))
		{
			func_duration   = 0;
			reduce_duration = 0;
			for (int i = 0; i < params.search_average_runs; ++i) {
				memset(params.match, 0, params.text_size*sizeof(int));
	
				// Call search function
				tic();
				algorithm.search(params);
				func_duration += toc(0) * 1000;
	
				tic();
				count = 0;
				for (unsigned long i = 0; i < params.text_size; i++)
					count += params.match[i];
				reduce_duration += toc(0) * 1000;
			}
			func_duration   /= params.search_average_runs;
			reduce_duration /= params.search_average_runs;
			printf(" %-10s | %-45s | %10d | %10.4f | %10.4f | %10.4f \n", 
				algorithm.tag, algorithm.name, count, func_duration, reduce_duration, func_duration + reduce_duration);
			alg_found = 1;
		}
	}
	return alg_found;
}

void exactmatch(const char* tags, struct search_parameters params){

	char** alg_list;
	int algs_len;
	parse_tagstring(&alg_list, &algs_len, tags);

	if (params.test_flags & GPU_TEST){
		printf("\n GPU Parallel Algorithms: \n\n");
		printf(" %-10s | %-45s | %10s | %10s | %10s | %10s | %10s | %10s \n", 
			"Tag", "Name", "Found", "Kernel", "Memcopy", 
			"Preprocess", "Reduction", "Total");
		for (int i = 0; i < algs_len; ++i)
			gpu_test(alg_list[i], params);
	}

	if (params.test_flags & CPU_TEST){
		printf("\n CPU Serial Algorithms: \n\n");
		printf(" %-10s | %-45s | %10s | %10s | %10s | %10s \n", 
			"Tag", "Name", "Found", "Kernel", "Reduction", "Total");
		for (int i = 0; i < algs_len; ++i)
			cpu_test(alg_list[i], params);
	}
	//Release Memory
	for (int i = 0; i < algs_len; ++i) free(alg_list[i]);
	free(alg_list);
}
