#ifndef COMMONS_H
#define COMMONS_H
#include <CL/cl.hpp>
typedef struct OpenCL {
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
} OpenCL;
#endif
