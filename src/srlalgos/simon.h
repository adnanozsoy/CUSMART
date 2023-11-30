#ifndef SIMON_H
#define SIMON_H

#include "util/parameters.h"

void simon(search_parameters params);

struct _cell{
	int element;
	struct _cell *next;
};
typedef struct _cell *List;

#endif
