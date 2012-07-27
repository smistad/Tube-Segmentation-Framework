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

    // Parse parameters from program arguments
    std::map<std::string, std::string> parameters = getParameters(argc, argv);
    std::string filename = argv[1];

    // Read dataset and transfer to device
    SIPL::int3 size;
    cl::Image3D dataset = readDatasetAndTransfer(ocl, filename, parameters, &size);

    // Run specified method on dataset
    TubeSegmentation TS;

    // Visualize result (and store)
    SIPL::Volume<SIPL::float3> * result = new SIPL::Volume<SIPL::float3>(size.x, size.y, size.z);
    for(int i = 0; i < result->getTotalSize(); i++) {
        SIPL::float3 v;
        v.x = TS.TDF[i];
        v.y = TS.centerline[i] ? 1.0:0.0;
        v.z = TS.segmentation[i] ? 1.0:0.0;
    }
    result->showMIP(SIPL::Y);

    return 0;
}
