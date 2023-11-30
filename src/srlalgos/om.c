/*
 * CUSMART: CUDA string matching algorithms research tool.
 * Copyright (C) 2019  CUSMART
 * Based on SMART project Copyright (C) 2012 Simone Faro and Thierry Lecroq
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>
 *
 * This is an implementation of the Optimal Mismatch algorithm
 * in D. M. Sunday.
 * A very fast substring search algorithm. Commun. ACM, vol.33, n.8, pp.132--142, (1990).
 */

#include "om.h"
#include "include/define.h"



typedef struct patternScanOrder { 
    int loc; 
    unsigned char c; 
} pattern; 

int freq[SIGMA]; 

void om_preQsBc(unsigned char *P, int m, int qbc[]) { 
    int i; 
    for (i=0;i<SIGMA;i++)   qbc[i]=m+1; 
    for (i=0;i<m;i++) qbc[P[i]]=m-i; 
} 

/* Construct an ordered pattern from a string. */ 
void orderPattern(unsigned char *x, int m, int (*pcmp)(), 
                  pattern *pat) { 
    int i; 
     
    for (i = 0; i <= m; ++i) { 
        pat[i].loc = i; 
        pat[i].c = x[i]; 
    } 
    qsort(pat, m, sizeof(pattern), pcmp); 
} 


/* Optimal Mismatch pattern comparison function. */ 
int optimalPcmp(pattern *pat1, pattern *pat2) { 
    float fx; 
     
    fx = freq[pat1->c] - freq[pat2->c]; 
    return(fx ? (fx > 0 ? 1 : -1) : 
           (pat2->loc - pat1->loc)); 
}


/* Find the next leftward matching shift for 
 the first ploc pattern elements after a 
 current shift or lshift. */ 
int matchShift(unsigned char *x, int m, int ploc, 
               int lshift, pattern *pat) { 
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
                  pattern *pat) { 
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



/* Optimal Mismatch string matching algorithm. */ 
void om(search_parameters params) { 
    int count = 0; 
    int i, j, adaptedGs[XSIZE], qsBc[SIGMA]; 
    pattern pat[XSIZE]; 
     
    /* Preprocessing */ 
    orderPattern(params.pattern, params.pattern_size, optimalPcmp, pat); 
    om_preQsBc(params.pattern, params.pattern_size, qsBc); 
    preAdaptedGs(params.pattern, params.pattern_size, adaptedGs, pat); 
     
    /* Searching */ 
    j = 0; 
    while (j <= params.text_size - params.pattern_size) { 
        i = 0; 
        while (i < params.pattern_size && pat[i].c == params.text[j + pat[i].loc]) 
            ++i; 
      if (i >= params.pattern_size) 
         params.match[j] = 1;
      j += MAX(adaptedGs[i],qsBc[params.text[j + params.pattern_size]]); 
    }
}
