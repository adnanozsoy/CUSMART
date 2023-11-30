
#ifndef TW_CUH
#define TW_CUH

#define MAX(a,b) ((a) > (b) ? (a) : (b))

__global__ 
void two_way1(unsigned char *text, unsigned long text_size,
	unsigned char *pattern, int pattern_size, int per, int ell, 
	int search_len, int *match);

__global__ 
void two_way2(unsigned char *text, unsigned long text_size,
	unsigned char *pattern, int pattern_size, int per, int ell, 
	int search_len, int *match);

int maxSuf(unsigned char *x, int m, int *p);
int maxSufTilde(unsigned char *x, int m, int *p);

#endif


