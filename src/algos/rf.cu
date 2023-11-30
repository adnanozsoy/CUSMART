
#include "rf.cuh"
#include "string"

__global__ void reverse_factor(unsigned char *text, unsigned long text_size, 
                            unsigned char *pattern, int pattern_size,
                            int *ttrans, unsigned char *tterminal,
                            int search_len, int *match){
	unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx;
	start_inx = thread_id * search_len;    
	unsigned long boundary = start_inx + search_len + pattern_size  - 1;
	boundary = boundary > text_size ? text_size : boundary;
	int i, j, shift, period, init, state;
	
	init = 0;
	period = pattern_size;
	j = start_inx;
	while (j < boundary && j <= text_size - pattern_size) {
      i = pattern_size - 1;
      state = init;
      shift = pattern_size;
      while (getTarget(state, text[i + j]) != UNDEFINED) {
         state = getTarget(state, text[i + j]);
         if (isTerminal(state)) {
            period = shift;
            shift = i;
         }
         --i;
      }
      if (i < 0) {
         match[j] = 1;
         shift = period;
      }
      j += shift;
   }

}




void buildSuffixAutomaton(unsigned char *x, int m, int *ttrans, int *tlength, int *tsuffix, unsigned char *tterminal) {
   int i, art, init, last, p, q, r, counter;
   unsigned char c;

   init = 0;
   art = 1;
   counter = 2;
   setSuffixLink(init, art);
   last = init;
   for (i = m-1; i >= 0; --i) {
      c = x[i];
      p = last;
      q = newState();
      setLength(q, getLength(p) + 1);
      while (p != init &&
             getTarget(p, c) == UNDEFINED) {
         setTarget(p, c, q);
         p = getSuffixLink(p);
      }
      if (getTarget(p, c) == UNDEFINED) {
         setTarget(init, c, q);
         setSuffixLink(q, init);
      }
      else
         if (getLength(p) + 1 == getLength(getTarget(p, c)))
            setSuffixLink(q, getTarget(p, c));
         else {
            r = newState();
            //copyVertex(r, getTarget(p, c));
            memcpy(ttrans+r*SIGMA, ttrans+getTarget(p, c)*SIGMA, SIGMA*sizeof(int));
            setSuffixLink(r, getSuffixLink(getTarget(p, c)));
            setLength(r, getLength(p) + 1);
            setSuffixLink(getTarget(p, c), r);
            setSuffixLink(q, r);
            while (p != art && getLength(getTarget(p, c)) >= getLength(r)) {
               setTarget(p, c, r);
               p = getSuffixLink(p);
            }
         }
      last = q;
   }
   setTerminal(last);
   while (last != init) {
      last = getSuffixLink(last);
      setTerminal(last);
   }
}
