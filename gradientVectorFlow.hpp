#ifndef GVF_H
#define GVF_H
#include "commons.hpp"
#include "SIPL/Types.hpp"
#include "parameters.hpp"
using namespace cl;

Image3D runGVF(OpenCL &ocl, Image3D * vectorField, paramList &parameters, SIPL::int3 &size, bool useLessMemory);

Image3D runFMGGVF(OpenCL &ocl, Image3D *vectorField, paramList &parameters, SIPL::int3 &size);

#endif
