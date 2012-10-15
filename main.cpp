#include "tube-segmentation.hpp"

#include <chrono>
#include <vector>
#define TIMING

#ifdef TIMING
#define INIT_TIMER auto timerStart = std::chrono::high_resolution_clock::now();
#define START_TIMER  timerStart = std::chrono::high_resolution_clock::now();
#define STOP_TIMER(name)  std::cout << "RUNTIME of " << name << ": " << \
        std::chrono::duration_cast<std::chrono::milliseconds>( \
                            std::chrono::high_resolution_clock::now()-timerStart \
                    ).count() << " ms " << std::endl; 
#else
#define INIT_TIMER
#define START_TIMER
#define STOP_TIMER(name)
#endif


int main(int argc, char ** argv) {
    INIT_TIMER
    START_TIMER
    OpenCL ocl; 

    if(argc == 1 || argv[1] == "--help") {
        // Print help message
        std::cout << "usage: " << argv[0] << " mhd-filename [options]" << std::endl << std::endl;
        std::cout << "available options: " << std::endl;
        std::vector<std::string> options = {
            "--device <type>", "which type of device to run calculations on (cpu|gpu)", "gpu",
            "--buffers-only", "disable writing to 3D images", "off",
            "--display", "display result using SIPL", "off",
            "--storage-dir <path>", "specify a directory to store the centerline and segmentation in", "off",
            "--minimum <value>", "set minimum threshold (if not specified it will find min automatically)", "auto",
            "--maximum <value>", "set maximum threshold (if not specified it will find min automatically)", "auto",
            "--mode <mode>", "look for black or white tubes (white|black)", "black",
            "--centerline-method", "specify which centerline method to use (ridge|gpu)", "gpu"
        };

        std::cout << "name\t\t\tdescription\t\t\t\t\t\tdefault value" << std::endl;
        for(int i = 0; i < options.size(); i += 3) {
            std::cout << options[i] << "\t" << options[i+1] << "\t" << options[i+2] << std::endl;
        }
        return 0;
    }

    ocl.context = createCLContextFromArguments(argc, argv);

    // Select first device
    cl::vector<cl::Device> devices = ocl.context.getInfo<CL_CONTEXT_DEVICES>();
    std::cout << "Using device: " << devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;
    ocl.queue = cl::CommandQueue(ocl.context, devices[0], CL_QUEUE_PROFILING_ENABLE);

    // Query the size of available memory
    unsigned int memorySize = devices[0].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    std::cout << "Available memory on selected device " << (double)memorySize/(1024*1024) << " MB "<< std::endl;

    // Parse parameters from program arguments
    paramList parameters = getParameters(argc, argv);
    std::string filename = argv[1];

    // Compile and create program
    if(parameters.count("buffers-only") == 0 && (int)devices[0].getInfo<CL_DEVICE_EXTENSIONS>().find("cl_khr_3d_image_writes") > -1) {
        ocl.program = buildProgramFromBinary(ocl.context, "kernels.cl");
        parameters["3d_write"] = "true";
    } else {
        ocl.program = buildProgramFromBinary(ocl.context, "kernels_no_3d_write.cl");
        std::cout << "Writing to 3D textures is not supported on the selected device." << std::endl;
    }

    SIPL::int3 size;
    TubeSegmentation TS;
    try {
        // Read dataset and transfer to device
        cl::Image3D dataset = readDatasetAndTransfer(ocl, filename, parameters, &size);

        // Run specified method on dataset
        if(parameters.count("centerline-method") && parameters["centerline-method"] == "ridge") {
            TS = runCircleFittingAndRidgeTraversal(ocl, dataset, size, parameters);
        } else {
            TS = runCircleFittingAndNewCenterlineAlg(ocl, dataset, size, parameters);
        }
    } catch(cl::Error e) {
        std::cout << "OpenCL error: " << getCLErrorString(e.err()) << std::endl;
        return 0;
    }
    ocl.queue.finish();
    STOP_TIMER("total")

    if(parameters.count("display") > 0) {
        // Visualize result
        SIPL::Volume<SIPL::float3> * result = new SIPL::Volume<SIPL::float3>(size.x, size.y, size.z);
        for(int i = 0; i < result->getTotalSize(); i++) {
            SIPL::float3 v;
            v.x = TS.TDF[i];
            v.y = 0;
            v.z = 0;
            v.y = TS.centerline[i] ? 1.0:0.0;
            v.z = TS.segmentation[i] ? 1.0:0.0;
            result->set(i,v);
        }
        result->showMIP(SIPL::Y);
    }
    if(parameters.count("display") > 0 || parameters.count("storage-dir") > 0 || parameters["centerline-method"] == "ridge") {
        // Cleanup transferred data
        delete[] TS.centerline;
        delete[] TS.segmentation;
        delete[] TS.TDF;
        if(parameters["centerline-method"] == "ridge")
            delete[] TS.radius;
    }

    return 0;
}
