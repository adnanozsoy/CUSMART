
#include "util/argtable3.h"
#include "util/parameters.h"
#include "exactmatch.h"
#include "wrappers/wrapper_list.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


void read_file(const char* file_path, unsigned char** text, unsigned long* text_len, int plen){
   FILE *f = fopen(file_path, "rb");
   fseek(f, 0, SEEK_END);
   *text_len = ftell(f);
   size_t alloc_len = *text_len + plen + 1;
   *text = (char*)malloc((alloc_len)*sizeof(char));
   rewind(f);
   fread(*text, sizeof(char), *text_len, f);
}

void print_parameters(search_parameters p){
   printf(" - stride length = %d\n", p.stride_length);
   printf(" - zero copy %s\n", p.pinned_memory ? "enabled" : "disabled");
   printf(" - gpu reduction %s\n", p.gpu_reduction ? "enabled" : "disabled");
   printf(" - cpu serial tests %s\n", p.test_flags & CPU_TEST ? "enabled" : "disabled");
}

int main(int argc, char *argv[])
{	
   int errorcode = 0;

   /* Setting up command-line arguments */
   struct arg_file *argv_infile = arg_file1(
      "f", "file", "<file>", "text file to perform search on");
   struct arg_str *argv_pattern = arg_str1(
      "p", "pattern", "<string>", "pattern text to search");
   struct arg_int *argv_block_size = arg_int0(
      "t", "stride-length", "<n>", "stride-length per thread (default: 2 x pattern-size)");
   struct arg_lit *argv_pinned_memory = arg_lit0(
      "i", "pinned-memory", "use pinned memory to reduce cpu to gpu memory transfers");
   struct arg_lit *argv_constant_memory = arg_lit0(
      "c", "constant-memory", "use constant memory to hold pattern");
   struct arg_str *argv_algorithms = arg_str0(
      "a", "algorithms", "<tag1>[,<tag2>]...", "selected algorithms to run (default: all)");
   struct arg_lit *argv_list_algorithms = arg_lit0(
      "l", "list-algorithms", "list all available algorithms");
   struct arg_lit *argv_enable_cpu_test = arg_lit0(
      "C", "test-cpu", "include cpu versions of algorithms in the test");
   struct arg_lit *argv_enable_gpu_test = arg_lit0(
      "G", "test-gpu", "include gpu versions of algorithms in the test");
   struct arg_lit *argv_gpu_reduction = arg_lit0(
      "r", "gpu-reduction", "Perform the counting of matched indexes on GPU side");
   struct arg_int *argv_block_dim = arg_int0(
      "b", "block-dimension", "<n>", "number of threads per block (default: 512)");
   struct arg_int *argv_average_runs = arg_int0(
      "x", "average-runs", "<n>", "number of repeated runs to calculate average runtime");
   struct arg_lit *argv_help = arg_lit0(
      NULL, "help", "display this help and exit");
   struct arg_end *argv_end = arg_end(20);
   void *argtable[] = {
      argv_infile, 
      argv_pattern, 
      argv_block_size,
      argv_pinned_memory, 
      argv_constant_memory,
      argv_algorithms, 
      argv_list_algorithms, 
      argv_enable_cpu_test,
      argv_enable_gpu_test,
      argv_gpu_reduction,
      argv_block_dim,
      argv_average_runs,
      argv_help, 
      argv_end};

   if (arg_nullcheck(argtable) != 0)
      printf("error: insufficient memory\n");

   /* Default values */
   argv_block_size->ival[0] = -1;
   argv_algorithms->sval[0] = "all";

   /* Parse arguments */
   int nerrors = arg_parse(argc, argv, argtable);

   /* Display help content*/
   if (argv_help->count > 0 || argc == 1){
      printf("Usage: cusmart ");
      arg_print_syntax(stdout, argtable, "\n\n");
      arg_print_glossary(stdout, argtable, "  %-40s %s\n");
      errorcode = 0;
      goto exit;
   }

   /* Display algorithm list*/
   if (argv_list_algorithms->count > 0)
   {
      printf(" List of Available Parallel Algorithms\n");
      printf("==============================\n");
      for (int i = 0; i < wrapper_list_len; ++i)
         printf("- [%-5s] %s\n", wrapper_list[i].tag, wrapper_list[i].name);
      errorcode = 0;
      goto exit;
   }

   /* If there are errors, display errors and exit */
   if (nerrors > 0){
      arg_print_errors(stdout, argv_end, "cusmart");
      errorcode = 1;
      goto exit;
   }

   struct search_parameters params = {0};
   const char *text_path = argv_infile->filename[0];
   // char *text_UA; /* unaligned text pointer */

   params.pattern_size = strlen(argv_pattern->sval[0]);
   params.pattern = calloc(params.pattern_size+1,sizeof(char));
   memcpy(params.pattern, argv_pattern->sval[0], params.pattern_size*sizeof(char));
   params.pinned_memory = 0;
   params.constant_memory = 0;
   params.gpu_reduction = 0;
   params.block_dim = 512;
   params.test_flags = 0;
   params.search_average_runs = 1;

   if (argv_block_size->count > 0)
      params.stride_length = argv_block_size->ival[0];
   else
      params.stride_length = params.pattern_size * 2;

   /* Set test option for cpu serial code */
   if (argv_enable_cpu_test->count > 0) {
      params.test_flags |= CPU_TEST;
   }
   if (argv_enable_gpu_test->count > 0) {
      params.test_flags |= GPU_TEST;
   }
   if (argv_enable_cpu_test->count == 0 && argv_enable_gpu_test->count == 0) {
      params.test_flags = (CPU_TEST | GPU_TEST);
   }

   /* Set pinned memory if the argument is passed */
   if (argv_pinned_memory->count > 0) {
      params.pinned_memory = 1;
   }

   /* Set constant memory flag*/
   if ( argv_constant_memory->count > 0) {
      params.constant_memory = 1;
   }

   /* Set gpu reduction flag */
   if (argv_gpu_reduction->count > 0) {
      params.gpu_reduction = 1;
   }

   /* Set block dimension */
   if (argv_block_dim->count > 0) {
      params.block_dim = argv_block_dim->ival[0];
   }

   if (argv_average_runs->count > 0) {
      params.search_average_runs = argv_average_runs->ival[0];
   }

   read_file(text_path, &params.text, &params.text_size, params.pattern_size);
   params.match = (int*)malloc(params.text_size * sizeof(int));

   printf("\n");
   printf("================\n");
   printf("Text file = %s (%lu characters)\n", text_path, params.text_size);
   printf("Search pattern = %s\n", params.pattern);
   printf("================\n");
   print_parameters(params);
   printf("\n");
   
   exactmatch(argv_algorithms->sval[0], params);
   
   free(params.match);
   free(params.pattern);
   free(params.text);
   exit:
   arg_freetable(argtable, sizeof(argtable) / sizeof(argtable[0]));
	return errorcode;
}
