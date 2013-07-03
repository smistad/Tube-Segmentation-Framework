#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include "commons.hpp"
#include "parameters.hpp"
using namespace cl;

Image3D runInverseGradientSegmentation(OpenCL &ocl, Image3D &centerline, Image3D &vectorField, Image3D &radius, SIPL::int3 size, paramList parameters);

Image3D runSphereSegmentation(OpenCL ocl, Image3D &centerline, Image3D &radius, SIPL::int3 size, paramList parameters);

#endif
