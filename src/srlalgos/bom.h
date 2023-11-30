#ifndef BOM_H
#define BOM_H

#include "util/parameters.h"

struct _cell{
    int element; 
    struct _cell *next;
  };
 
typedef struct _cell *List;

int getTransition(unsigned char *x, int p, List L[], unsigned char c);
void setTransition(int p, int q, List L[]);
void oracle(unsigned char *x, int m, char T[], List L[]);
void bom(search_parameters);

#endif
