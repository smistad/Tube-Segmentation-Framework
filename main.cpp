#include "tube-segmentation.hpp"
#include "SIPL/Core.hpp"
#include "tsf-config.h"


int main(int argc, char ** argv) {
    if(argc == 1 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        // Print help message
        std::cout << std::endl;
        std::cout << "Tube Segmentation Framework" << std::endl;
        std::cout << "===========================" << std::endl;
        std::cout << "Copyright Erik Smistad 2013 - See file LICENSE for license information." << std::endl;
        std::cout << std::endl;
        std::cout << "Usage: " << argv[0] << " inputFilename.mhd <parameters>" << std::endl;
        std::cout << std::endl;
        std::cout << "Example: " << argv[0] << " tests/data/synthetic/dataset_1/noisy.mhd --parameters Synthetic-Vascusynth --display" << std::endl;
        std::cout << std::endl;
        std::cout << "Available parameter presets: " << std::endl;
        std::cout << "* Lung-Airways-CT" << std::endl;
        std::cout << "* Neuro-Vessels-USA" << std::endl;
        std::cout << "* Neuro-Vessels-MRA" << std::endl;
        std::cout << "* AAA-Vessels-CT" << std::endl;
        std::cout << "* Liver-Vessels-CT" << std::endl;
        std::cout << "* Synthetic-Vascusynth" << std::endl;
        std::cout << std::endl;
        std::cout << "The parameter preset is set with the program argument \"--parameters <name>\"." << std::endl;
        std::cout << std::endl;
        std::cout << "Available parameters: " << std::endl;
        printAllParameters();
        exit(-1);
    }

    // Load default parameters and parse parameters from program arguments
    paramList parameters = getParameters(argc, argv);
    std::string filename = argv[1];


    TSFOutput * output;
    try {
		output = run(filename, parameters, std::string(KERNELS_DIR));
    } catch(SIPL::SIPLException &e) {
    	std::cout << e.what() << std::endl;

    	return -1;
    }

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
