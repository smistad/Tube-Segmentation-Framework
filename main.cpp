#include "tube-segmentation.hpp"
#include <fstream>
#include "SIPL/Core.hpp"


#include <vector>

//#define TIMING

#ifdef TIMING
#include <chrono>
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
    OpenCL ocl; 

    if(argc == 1 || argv[1] == "--help") {
        // Print help message
        std::cout << "usage: " << argv[0] << " mhd-filename [options]" << std::endl << std::endl;
        std::cout << "available options: " << std::endl;

		std::cout << "name\t description [default value]" << std::endl;
        std::cout << "--device <type>\t which type of device to run calculations on (cpu|gpu) [gpu]" << std::endl;
		std::cout << "--buffers-only\t disable writing to 3D images [off]" << std::endl;
        std::cout << "--display\t display result using SIPL [off]" << std::endl;
        std::cout << "--storage-dir <path>\t specify a directory to store the centerline and segmentation in [off]" << std::endl;
        std::cout << "--minimum <value>\t set minimum threshold (if not specified it will find min automatically) [auto]" << std::endl;
        std::cout << "--maximum <value>\t set maximum threshold (if not specified it will find min automatically) [auto]" << std::endl;
        std::cout << "--mode <mode>\t look for black or white tubes (white|black) [black]" << std::endl;
        std::cout << "--centerline-method\t specify which centerline method to use (ridge|gpu) [gpu]" << std::endl;
        std::cout << "--no-segmentation\t turns off segmentation and returns centerline only" << std::endl;

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

    // Check if parameters is set
    if(getParamStr(parameters, "parameters") != "none") {
    	std::string parameterFilename;
    	if(getParamStr(parameters, "centerline-method") == "gpu") {
    		parameterFilename = "parameters/centerline-gpu/" + getParamStr(parameters, "parameters");
    	} else if(getParamStr(parameters, "centerline-method") == "ridge") {
    		parameterFilename = "parameters/centerline-ridge/" + getParamStr(parameters, "parameters");
    	}
    	std::cout << parameterFilename << std::endl;
    	if(parameterFilename.size() > 0) {
    		// Load file and parse parameters
    		std::ifstream file(parameterFilename.c_str());
    		if(!file.is_open()) {
    			std::cout << "ERROR: could not open parameter file " << parameterFilename << std::endl;
    			exit(-1);
    		}

    		std::string line;
    		while(!file.eof()) {
				getline(file, line);
				if(line.size() == 0)
					continue;
    			// split string on the first space
    			int spacePos = line.find(" ");
    			if(spacePos != std::string::npos) {
    				// parameter with value
					std::string name = line.substr(0, spacePos);
					std::string value = line.substr(spacePos+1);
					parameters = setParameter(parameters, name, value);
    			} else {
    				// parameter with no value
    				parameters = setParameter(parameters, line, "true");
    			}
    		}
    		file.close();
    	}
    }

    /*
    // Write out parameter list
    std::cout << "The following parameters are set: " << std::endl;
    unordered_map<std::string, std::string>::iterator it;
    for(it = parameters.begin(); it != parameters.end(); it++) {
    	std::cout << it->first << " " << it->second << std::endl;
    }
    */

    // Compile and create program
    if(!getParamBool(parameters, "buffers-only") && (int)devices[0].getInfo<CL_DEVICE_EXTENSIONS>().find("cl_khr_3d_image_writes") > -1) {
        ocl.program = buildProgramFromSource(ocl.context, "kernels.cl");
        BoolParameter v = parameters.bools["3d_write"];
        v.set(true);
        parameters.bools["3d_write"] = v;
    } else {
        BoolParameter v = parameters.bools["3d_write"];
        v.set(false);
        parameters.bools["3d_write"] = v;
        ocl.program = buildProgramFromSource(ocl.context, "kernels_no_3d_write.cl");
        std::cout << "Writing to 3D textures is not supported on the selected device." << std::endl;
    }

    START_TIMER
    SIPL::int3 size;
    TubeSegmentation TS;
    try {
        // Read dataset and transfer to device
        cl::Image3D dataset = readDatasetAndTransfer(ocl, filename, parameters, &size);

        // Run specified method on dataset
        if(getParamStr(parameters, "centerline-method") == "ridge") {
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

    if(getParamBool(parameters, "display")) {
        // Visualize result
        SIPL::Volume<SIPL::float3> * result = new SIPL::Volume<SIPL::float3>(size.x, size.y, size.z);
        bool tdfOnly = getParamBool(parameters, "tdf-only");
        bool noSegmentation = getParamBool(parameters, "no-segmentation");
        for(int i = 0; i < result->getTotalSize(); i++) {
            SIPL::float3 v;
            v.x = TS.TDF[i];
            v.y = 0;
            v.z = 0;
            if(!tdfOnly)
				v.y = TS.centerline[i] ? 1.0:0.0;
            if(!noSegmentation && !tdfOnly)
                v.z = TS.segmentation[i] ? 1.0:0.0;
            result->set(i,v);
        }
        result->showMIP(SIPL::Y);
    }
    if(getParamBool(parameters, "display") || getParamStr(parameters, "storage-dir") != "off" || getParamStr(parameters, "centerline-method") == "ridge") {
        // Cleanup transferred data
		if(getParamBool(parameters, "tdf-only")) {
			delete[] TS.TDF;
    	} else {
			delete[] TS.centerline;
			delete[] TS.TDF;
			if(!getParamBool(parameters, "no-segmentation"))
				delete[] TS.segmentation;
			if(getParamStr(parameters, "centerline-method") == "ridge")
				delete[] TS.radius;
    	}
    }
    return 0;
}
