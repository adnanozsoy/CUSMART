#include "trf.cuh"

__global__ void turbo_reverse_factor(unsigned char *text, unsigned long text_size, 
			       unsigned char *pattern, int pattern_size, int *ttrans,
			       unsigned char *tterminal, int *tshift, int *mpNext, int search_len, int *match){
	unsigned long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long start_inx = thread_id * search_len;    
	unsigned long boundary = start_inx + search_len + pattern_size  - 1;
	boundary = boundary > text_size ? text_size : boundary;
	int period, i, j, shift, u, periodOfU, disp, init, state, mu, m;
	m = pattern_size;
	init = 0;
	period = pattern_size - mpNext[pattern_size];
	i = 0;
	shift = pattern_size;
	
	while (m && *pattern && (*pattern == *text)){
	  ++pattern;
	  ++text;
	  --m;
	}
	if (m == 0){
	  match[0] = 1;
	}
	//	if (strncmp(x, y, m) == 0)
	//  OUTPUT(0);
	j = start_inx;
	while (j < boundary  && j <= (text_size - pattern_size)) {
	  i = pattern_size-1;
	  state = init;
	  u = pattern_size-1-shift;
	  periodOfU = (shift != pattern_size ?  pattern_size - shift - mpNext[m - shift] : 0);
	  shift = pattern_size;
	  disp = 0;
	  while (i > u && getTarget(state, text[i + j]) != UNDEFINED) {
	    disp += getShift(state, text[i + j]);
	    state = getTarget(state, text[i + j]);
	    if (isTerminal(state))
	      shift = i;
	    --i;
	  }
	  if (i <= u)
	    if (disp == 0) {
	      match[j] = 1;
	      shift = period;
	    }
	    else {
	      mu = (u + 1)/2;
	      if (periodOfU <= mu) {
		u -= periodOfU;
		while (i > u && getTarget(state, text[i + j]) != UNDEFINED) {
                  disp += getShift(state, text[i + j]);
                  state = getTarget(state, text[i + j]);
                  if (isTerminal(state))
		    shift = i;
                  --i;
		}
		if (i <= u)
                  shift = disp;
	      }
	      else {
		u = u - mu - 1;
		while (i > u && getTarget(state, text[i + j]) != UNDEFINED) {
                  disp += getShift(state, text[i + j]);
                  state = getTarget(state, text[i + j]);
                  if (isTerminal(state))
		    shift = i;
                  --i;
		}
	      }
	    }
	  j += shift;
	}
}

void preMpforTRF(unsigned char *x, int m, int mpNext[]) {
   int i, j;
   i = 0;
   j = mpNext[0] = -1;
   while (i < m) {
      while (j > -1 && x[i] != x[j])
         j = mpNext[j];
      mpNext[++i] = ++j;
   }
}

void buildSuffixAutomaton4TRF(unsigned char *x, int m, 
   int *ttrans, int *tlength, int *tposition, int *tsuffix, unsigned char *tterminal, int *tshift) {
   int i, art, init, last, p, q, r, counter, tmp;
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
      setPosition(q, getPosition(p) + 1);
      while (p != init &&
             getTarget(p, c) == UNDEFINED) {
         setTarget(p, c, q);
         setShift(p, c, getPosition(q) - getPosition(p) - 1);
         p = getSuffixLink(p);
      }
      if (getTarget(p, c) == UNDEFINED) {
         setTarget(init, c, q);
         setShift(init, c, getPosition(q) - getPosition(init) - 1);
         setSuffixLink(q, init);
      }
      else
         if (getLength(p) + 1 == getLength(getTarget(p, c)))
            setSuffixLink(q, getTarget(p, c));
         else {
            r = newState();
            tmp = getTarget(p, c);
            memcpy(ttrans+r*SIGMA, ttrans+tmp*SIGMA, SIGMA*sizeof(int));
            memcpy(tshift+r*SIGMA, tshift+tmp*SIGMA, SIGMA*sizeof(int));
            setPosition(r, getPosition(tmp));
            setSuffixLink(r, getSuffixLink(tmp));
            setLength(r, getLength(p) + 1);
            setSuffixLink(tmp, r);
            setSuffixLink(q, r);
            while (p != art && getLength(getTarget(p, c)) >= getLength(r)) {
               setShift(p, c, getPosition(getTarget(p, c)) - getPosition(p) - 1);
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
