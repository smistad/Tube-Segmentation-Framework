#ifndef COMMONS_H
#define COMMONS_H
#include <CL/cl.hpp>
typedef struct OpenCL {
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
} OpenCL;
static inline float log2(double a) {
	return log(a)/log(2.0);
}

static inline float round(float a) {
	return floor(a+0.5);
}
#endif
