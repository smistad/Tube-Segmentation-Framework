#include "commons.hpp"
#include "SIPL/Types.hpp"
using namespace cl;

void runSplineTDF(
        OpenCL &ocl,
        SIPL::int3 &size,
        Image3D *vectorField,
        Buffer *TDF,
        Buffer *radius,
        float radiusMin,
        float radiusMax,
        float radiusStep
        );
void runCircleFittingTDF(OpenCL &ocl, SIPL::int3 &size, Image3D * vectorField, Buffer * TDF, Buffer * radius, float radiusMin, float radiusMax, float radiusStep,bool useMask,char * mask);
void runVesselnessTDF(OpenCL &ocl, SIPL::int3 &size, Image3D * vectorField, Buffer * TDF);
