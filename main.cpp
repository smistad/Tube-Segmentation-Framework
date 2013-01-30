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

    // Parse parameters from program arguments
    paramList parameters = getParameters(argc, argv);
    std::string filename = argv[1];
    TSFOutput * output;
    try {
		output = run(filename, parameters, argc, argv);
    } catch(SIPL::SIPLException e) {
    	std::cout << e.what() << std::endl;
    }

    SIPL::int3 * size = output->getSize();

    if(getParamBool(parameters, "display")) {
        // Visualize result
        SIPL::Volume<SIPL::float3> * result = new SIPL::Volume<SIPL::float3>(size->x, size->y, size->z);
        float * TDF;
        char * centerline;
        char * segmentation;
        if(output->hasTDF())
        	TDF = output->getTDF();
        if(output->hasCenterlineVoxels())
        	centerline = output->getCenterlineVoxels();
        if(output->hasSegmentation())
        	segmentation = output->getSegmentation();
        for(int i = 0; i < result->getTotalSize(); i++) {
            SIPL::float3 v;
            if(output->hasTDF())
            	v.x = TDF[i];
            if(output->hasCenterlineVoxels())
				v.y = centerline[i] ? 1.0:0.0;
            if(output->hasSegmentation())
                v.z = segmentation[i] ? 1.0:0.0;
            result->set(i,v);
        }
        result->showMIP(SIPL::Y);
    }
    // free data
    output->~TSFOutput();
    /*
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
    */
    return 0;
}
