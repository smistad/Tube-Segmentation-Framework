#include "tubeDetectionFilters.hpp"
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
    ) {

    /*
    // Create blending functions
    int samples = 3;
    float s = 0.5;
    float * blendingFunctions = new float[4*samples]; // 4 * samples per arm
    for(int i = 0; i < samples; i++) {
        float u = (float)i / (samples-1);
        blendingFunctions[i*4] = -s*u*u*u + 2*s*u*u - s*u;
        blendingFunctions[i*4+1] = (2-s)*u*u*u + (s-3)*u*u + 1;
        blendingFunctions[i*4+2] = (s-2)*u*u*u + (3-2*s)*u*u + s*u;
        blendingFunctions[i*4+3] = s*u*u*u - s*u*u;
    }

    // Transfer to device
    Buffer bufferBlendingFunctions = Buffer(
     ocl.context,
     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
     sizeof(float)*samples*4,
     blendingFunctions
    );
    */

    Kernel TDFKernel(ocl.program, "splineTDF");
    TDFKernel.setArg(0, *vectorField);
    TDFKernel.setArg(1, *TDF);
    TDFKernel.setArg(2, std::max(1.0f, radiusMin));
    TDFKernel.setArg(3, radiusMax);
    TDFKernel.setArg(4, radiusStep);
    //TDFKernel.setArg(5, bufferBlendingFunctions);
    TDFKernel.setArg(5, 12); // arms
    //TDFKernel.setArg(6, samples); // samples per arm
    TDFKernel.setArg(6, *radius);
    TDFKernel.setArg(7, 0.1f);

    ocl.queue.enqueueNDRangeKernel(
            TDFKernel,
            NullRange,
            NDRange(size.x,size.y,size.z),
            NDRange(4,4,4)
    );
}

void runCircleFittingTDF(OpenCL &ocl, SIPL::int3 &size, Image3D * vectorField, Buffer * TDF, Buffer * radius, float radiusMin, float radiusMax, float radiusStep) {
    Kernel circleFittingTDFKernel(ocl.program, "circleFittingTDF");
    circleFittingTDFKernel.setArg(0, *vectorField);
    circleFittingTDFKernel.setArg(1, *TDF);
    circleFittingTDFKernel.setArg(2, *radius);
    circleFittingTDFKernel.setArg(3, radiusMin);
    circleFittingTDFKernel.setArg(4, radiusMax);
    circleFittingTDFKernel.setArg(5, radiusStep);

    ocl.queue.enqueueNDRangeKernel(
            circleFittingTDFKernel,
            NullRange,
            NDRange(size.x,size.y,size.z),
            NDRange(4,4,4)
    );

}

