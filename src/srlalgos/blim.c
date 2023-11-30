/*
 * SMART: string matching algorithms research tool.
 * Copyright (C) 2012  Simone Faro and Thierry Lecroq
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
 * contact the authors at: faro@dmi.unict.it, thierry.lecroq@univ-rouen.fr
 * download the tool at: http://www.dmi.unict.it/~faro/smart/
 *
 * This is an implementation of the Bit Parallel Length Invariant Matcher
 * in M. O. Kulekci. 
 * A method to overcome computer word size limitation in bit-parallel pattern matching. 
 * Proceedings of the 19th International Symposium on Algorithms and Computation, ISAAC 2008, 
 * Lecture Notes in Computer Science, vol.5369, pp.496--506, Springer-Verlag, Berlin, Gold Coast, Australia, (2008).
 */

#include "include/define.h"
#include "blim.h"

#include "stdlib.h"
#include "string.h"
#define MAXWSIZE (XSIZE + WORD)

void blim(search_parameters params)
{
   int i,j,k;
   unsigned int   wsize = WORD - 1 + params.pattern_size;
   unsigned long* MM = (unsigned long*)malloc(sizeof(unsigned long)*SIGMA * MAXWSIZE);
   unsigned long  tmp, F;
   unsigned int   ScanOrder[XSIZE];
   unsigned int   MScanOrder[XSIZE];
   unsigned int*  so  = ScanOrder;
   unsigned int*  mso = MScanOrder;
   unsigned int   shift[SIGMA];

   /* Preprocessing */
   memset(MM,0xff,sizeof(unsigned long)*SIGMA*wsize);
    for(i=0;i<WORD;i++){
      tmp = 1 << i;
      for(j=0;j<params.pattern_size;j++){
         for(k=0;k<SIGMA;k++) MM[((i+j)*SIGMA) + k] &= ~tmp;
         MM[ params.pattern[j] + ((i+j)*SIGMA) ]|= tmp;
      }
   }
   
   for(i=0;i<SIGMA;i++) shift[i] = wsize + 1;
   for(i=0;i<params.pattern_size;i++) shift[params.pattern[i]] = wsize - i;
   
   for(i=params.pattern_size-1;i>=0;i--){
      k=i;
      while (k<wsize){
         *so=k;
         *mso = SIGMA*k;
         so++;
         mso++;
         k+=params.pattern_size;
      }
   }
   
   /* Searching */
   i = 0;
   F = MM[MScanOrder[0]+params.text[i+ScanOrder[0]]] & MM[MScanOrder[1]+params.text[i+ScanOrder[1]]];
   while(i<params.text_size) {
      for(j=2;F && j<wsize;j++){ 
         F &= MM[MScanOrder[j]+params.text[i+ScanOrder[j]]]; 
      }
      if (F) {
         for(j=0;j<WORD;j++) 
            if (F & (1<<j)) 
               if(i+j<=params.text_size-params.pattern_size) params.match[i+j] = 1;
      }
      i+=shift[params.text[i+wsize]];
      F = MM[MScanOrder[0]+params.text[i+ScanOrder[0]]] & MM[MScanOrder[1]+params.text[i+ScanOrder[1]]];
   }
}
