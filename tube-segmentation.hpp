#ifndef TUBE_SEGMENTATION
#define TUBE_SEGMENTATION

#include "OpenCLUtilities/openCLUtilities.hpp"
#include "SIPL/Core.hpp"
#include <iostream>

typedef struct OpenCL {
    cl::Context &context;
    cl::CommandQueue &queue;
    cl::Program &program;
} OpenCL;


#endif
