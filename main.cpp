#include "tube-segmentation.hpp"


int main(int argc, char ** argv) {
    OpenCL ocl; 

    ocl.context = createCLContextFromArguments(argc, argv);

    // Select first device
    cl::vector<cl::Device> devices = ocl.context.getInfo<CL_CONTEXT_DEVICES>();
    std::cout << "Using device: " << devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;
    ocl.queue = cl::CommandQueue(ocl.context, devices[0]);

    // Query the size of available memory
    unsigned int memorySize = devices[0].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    std::cout << "Available memory on selected device " << (double)memorySize/(1024*1024) << " MB "<< std::endl;

    // Compile and create program
    ocl.program = buildProgramFromSource(ocl.context, "kernels.cl");

    // Read dataset and transfer to device
    readDatasetAndTransfer(ocl, filename, parameters);

    // Do cropping if required

    // Run specified method

    // Visualize result (and store)
    SIPL::Volume<SIPL::float3> * result = SIPL::Volume<SIPL::float3>();
    for(int i = 0; i < result->getTotalSize(); i++) {
        SIPL::float3 v;
        v.x = T.tdf[i];
        v.y = T.centerline[i];
        v.z = T.segmentation[i];
    }
    result->showMIP(SIPL::Y);

    return 0;
}
