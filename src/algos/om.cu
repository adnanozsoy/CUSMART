#include "om.cuh"
#include <stdlib.h>

int freq[SIGMA];

__global__ void optimal_mismatch(unsigned char *text, unsigned long text_size,
	unsigned char *pattern, int pattern_size,int *adaptedGs, int *qsBc, 
	ompattern *pat, int search_len, int *match)
{
	unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx = thread_id * search_len;
	
	unsigned long boundary = start_inx + search_len + pattern_size - 1;
	boundary = boundary > text_size ? text_size : boundary;
	int i;
	unsigned long j = start_inx;

	while (j < boundary && j <= text_size - pattern_size) {
		i=0;
		while (i < pattern_size && pat[i].c == text[j + pat[i].loc])
			++i;
		
		if (i >= pattern_size)
			match[j] = 1;
		
		j += MAX(adaptedGs[i],qsBc[text[j + pattern_size]]);
	}
}


/* Optimal Mismatch pattern comparison function. */ 
int optimalPcmp(const void* pa, const void* pb) { 
    float fx; 
    ompattern* pat1 = (ompattern*)pa;
    ompattern* pat2 = (ompattern*)pb;
    fx = freq[pat1->c] - freq[pat2->c]; 
    return (fx ? (fx > 0 ? 1 : -1) : (pat2->loc - pat1->loc)); 
}


void om_preQsBc(unsigned char *P, int m, int qbc[]) { 
    int i; 
    for (i=0;i<SIGMA;i++)   qbc[i]=m+1; 
    for (i=0;i<m;i++) qbc[P[i]]=m-i; 
}


/* Construct an ordered pattern from a string. */ 
void orderPattern(unsigned char *x, int m, int (*pcmp)(const void*, const void*), ompattern *pat) { 
    int i;      
    for (i = 0; i <= m; ++i) { 
        pat[i].loc = i; 
        pat[i].c = x[i]; 
    }
    qsort(pat, m, sizeof(ompattern), pcmp); 
}


/* Find the next leftward matching shift for 
 the first ploc pattern elements after a 
 current shift or lshift. */ 
int matchShift(unsigned char *x, int m, int ploc, 
               int lshift, ompattern *pat) { 
    int i, j; 
     
    for (; lshift < m; ++lshift) { 
        i = ploc; 
        while (--i >= 0) { 
            if ((j = (pat[i].loc - lshift)) < 0) 
                continue; 
            if (pat[i].c != x[j]) 
                break; 
        } 
        if (i < 0) 
            break; 
    } 
    return(lshift); 
}


/* Constructs the good-suffix shift table 
 from an ordered string. */ 
void preAdaptedGs(unsigned char *x, int m, int adaptedGs[], 
                  ompattern *pat) { 
    int lshift, i, ploc; 
     
    adaptedGs[0] = lshift = 1; 
    for (ploc = 1; ploc <= m; ++ploc) { 
        lshift = matchShift(x, m, ploc, lshift, pat); 
        adaptedGs[ploc] = lshift; 
    } 
    for (ploc = 0; ploc <= m; ++ploc) { 
        lshift = adaptedGs[ploc]; 
        while (lshift < m) { 
            i = pat[ploc].loc - lshift; 
            if (i < 0 || pat[ploc].c != x[i]) 
                break; 
            ++lshift; 
            lshift = matchShift(x, m, ploc, lshift, pat); 
        } 
        adaptedGs[ploc] = lshift; 
    } 
}



void om_preprocess(unsigned char *x, int m, ompattern *pat, int qsBc[], int adaptedGs[]){
	orderPattern(x, m, optimalPcmp, pat); 
	om_preQsBc(x, m, qsBc); 
	preAdaptedGs(x, m, adaptedGs, pat);
}
