#ifndef TUBE_SEGMENTATION
#define TUBE_SEGMENTATION

#include "OpenCLUtilities/openCLUtilities.hpp"
#include "SIPL/Core.hpp"
#include <iostream>
#include <string>
#include <map>

typedef struct OpenCL {
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
} OpenCL;

typedef struct TubeSegmentation {
    float *Fx, *Fy, *Fz; // The GVF vector field
    float *TDF; // The TDF response
    bool *centerline;
    bool *segmentation;
} TubeSegmentation;

cl::Image3D readDatasetAndTransfer(OpenCL, std::string, std::map<std::string, std::string>, SIPL::int3 *);

std::map<std::string, std::string> getParameters(int argc, char ** argv);

TubeSegmentation runCircleFittingMethod(OpenCL, cl::Image3D dataset, std::map<std::string, std::string> parameters);

#endif
