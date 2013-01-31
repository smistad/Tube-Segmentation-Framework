#include "tube-segmentation.hpp"
#include "SIPL/Core.hpp"

int main(int argc, char ** argv) {

    // Load default parameters and parse parameters from program arguments
    paramList parameters = getParameters(argc, argv);
    std::string filename = argv[1];
    TSFOutput * output;
    try {
		output = run(filename, parameters);
    } catch(SIPL::SIPLException e) {
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
