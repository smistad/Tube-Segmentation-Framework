#ifndef TUBE_SEGMENTATION
#define TUBE_SEGMENTATION

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#include "OpenCLUtilities/openCLUtilities.hpp"
#include "SIPL/Types.hpp"
#include <iostream>
#include <string>
#include <set>
#ifdef CPP11
#include <unordered_map>
using std::unordered_map;
#else
#include <boost/unordered_map.hpp>
using boost::unordered_map;
#endif
#include "commons.hpp"
#include "parameters.hpp"
#include "SIPL/Exceptions.hpp"
#include "inputOutput.hpp"

typedef struct TubeSegmentation {
    float *Fx, *Fy, *Fz; // The GVF vector field
    float *FxSmall, *FySmall, *FzSmall; // The GVF vector field
    float *TDF; // The TDF response
    float *radius;
    char *centerline;
    char *segmentation;
    float *intensity;
} TubeSegmentation;


/*
 * For debugging.
 */
void print(paramList parameters);

cl::Image3D readDatasetAndTransfer(OpenCL &ocl, std::string, paramList &parameters, SIPL::int3 *, TSFOutput *);

void runCircleFittingAndRidgeTraversal(OpenCL *, cl::Image3D *dataset, SIPL::int3 * size, paramList &parameters, TSFOutput *);

void runCircleFittingAndNewCenterlineAlg(OpenCL *, cl::Image3D *dataset, SIPL::int3 * size, paramList &parameters, TSFOutput *);

void runCircleFittingAndTest(OpenCL *, cl::Image3D *dataset, SIPL::int3 * size, paramList &parameters, TSFOutput *);


TSFOutput * run(std::string filename, paramList &parameters, std::string kernel_dir);

#endif
