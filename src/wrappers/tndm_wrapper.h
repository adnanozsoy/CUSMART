#ifndef TNDM_WRAPPER_H
#define TNDM_WRAPPER_H

#include "util/parameters.h"

#define SIGMA 256

extern "C" search_info two_way_nondeterministic_dawg_wrapper(search_parameters);

#endif
