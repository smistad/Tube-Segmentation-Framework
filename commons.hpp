#ifndef COMMONS_H
#define COMMONS_H
#include <CL/cl.hpp>
typedef struct OpenCL {
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
} OpenCL;

#ifdef WIN32
// Add some math functions that are missing from the windows math library
template <class T>
static inline double log2(T a) {
	return log((double)a)/log(2.0);
}

template <class T>
static inline double round(T a) {
	return floor((double)a+0.5);
}
#endif

#endif
