#include "tube-segmentation.hpp"
#include "SIPL/Core.hpp"
#include "tsf-config.h"
#ifdef CPP11
#include <chrono>
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
#endif

int main(int argc, char ** argv) {

    // Load default parameters and parse parameters from program arguments
    paramList parameters = getParameters(argc, argv);
    std::string filename = argv[1];


#ifdef CPP11
    high_resolution_clock::time_point timerStart;
    if(getParamBool(parameters, "timer-total")) {
		timerStart = high_resolution_clock::now();
    }
#endif
    TSFOutput * output;
    try {
		output = run(filename, parameters, std::string(KERNELS_DIR));
    } catch(SIPL::SIPLException &e) {
    	std::cout << e.what() << std::endl;

    	return -1;
    }

#ifdef CPP11
    if(getParamBool(parameters, "timer-total")) {
		std::cout << "TOTAL RUNTIME: " <<
				duration_cast<milliseconds>(
				high_resolution_clock::now()-timerStart).count() <<
				" ms " << std::endl;
    }
#endif

    if(getParamBool(parameters, "display")) {
        // Visualize result
		SIPL::int3 * size = output->getSize();
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

    return 0;
}
