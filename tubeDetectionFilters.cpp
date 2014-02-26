#include "tubeDetectionFilters.hpp"
using namespace cl;

void runVesselnessTDF(
        OpenCL &ocl, 
        SIPL::int3 &size, 
        Image3D * vectorField, 
        Buffer * TDF
    ) {
    Kernel kernel(ocl.program, "vesselnessTDF");
    kernel.setArg(0, *vectorField);
    kernel.setArg(1, *TDF);
    kernel.setArg(2, 0.5f);
    kernel.setArg(3, 0.5f);
    kernel.setArg(4, 100.0f);
    ocl.queue.enqueueNDRangeKernel(
            kernel,
            NullRange,
            NDRange(size.x,size.y,size.z),
            NDRange(4,4,4)
    );
}

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

void runCircleFittingTDF(OpenCL &ocl, SIPL::int3 &size, Image3D * vectorField, Buffer * TDF, Buffer * radius, float radiusMin, float radiusMax, float radiusStep, bool useMask, char * mask) {
    Kernel circleFittingTDFKernel(ocl.program, "circleFittingTDF");
    Image3D clMask;
    if(useMask) {
        // Transfer mask to GPU
        std::cout << "using mask" << std::endl;
        std::cout << size.x << " " << size.y << " " << size.z << std::endl;
        std::cout << (int)mask[0] << std::endl;
        clMask = Image3D(
                ocl.context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                ImageFormat(CL_R, CL_SIGNED_INT8),
                size.x, size.y, size.z,
                0, 0,
                mask
        );
        circleFittingTDFKernel.setArg(7, clMask);
    } else {
        // Setting dummy image if mask is not to be used. If not it will give error
        circleFittingTDFKernel.setArg(7, *vectorField);
    }
    circleFittingTDFKernel.setArg(0, *vectorField);
    circleFittingTDFKernel.setArg(1, *TDF);
    circleFittingTDFKernel.setArg(2, *radius);
    circleFittingTDFKernel.setArg(3, radiusMin);
    circleFittingTDFKernel.setArg(4, radiusMax);
    circleFittingTDFKernel.setArg(5, radiusStep);
    char useMaskValue = useMask ? 1 : 0;
    circleFittingTDFKernel.setArg(6, useMaskValue);

    ocl.queue.enqueueNDRangeKernel(
            circleFittingTDFKernel,
            NullRange,
            NDRange(size.x,size.y,size.z),
            NDRange(4,4,4)
    );

}

