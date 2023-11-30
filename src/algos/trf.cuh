#ifndef TRF_CUH
#define TRF_CUH

#include "include/define.cuh"

#define UNDEFINED       -1
#define getTarget(p, c) ttrans[(p)*SIGMA+(c)]
#define isTerminal(p) tterminal[(p)]
#define setPosition(p,q) tposition[(p)]=(q)
#define getPosition(p) tposition[(p)]
#define setShift(p,c,q) tshift[(p)*SIGMA + c] = (q)
#define getShift(p,c) tshift[(p)*SIGMA + c]
#define setTarget(p, c, q) ttrans[(p)*SIGMA+(c)] = (q)
#define setSuffixLink(p, q) tsuffix[(p)] = (q)
#define getSuffixLink(p) tsuffix[(p)]
#define newState() counter++
#define setLength(p, q) tlength[(p)] = (q)
#define getLength(p) tlength[(p)]
#define setTerminal(p) tterminal[(p)] = 1

__global__ 
void turbo_reverse_factor(unsigned char *text, unsigned long text_size, 
			  unsigned char *pattern, int pattern_size, int *ttrans,
			  unsigned char *tterminal, int *tshift, int *mpNext,
			  int search_len, int *match);
void buildSuffixAutomaton4TRF(unsigned char *x, int m, 
			      int *ttrans, int *tlength, int *tposition,
			      int *tsuffix, unsigned char *tterminal, int *tshift);
void preMpforTRF(unsigned char *x, int m, int mpNext[]);

#endif
