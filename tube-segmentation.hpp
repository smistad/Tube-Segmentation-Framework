#ifndef TUBE_SEGMENTATION
#define TUBE_SEGMENTATION

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#include "OpenCLUtilities/openCLUtilities.hpp"
#include "SIPL/Types.hpp"
#include <iostream>
#include <string>
#ifdef CPP11
#include <unordered_map>
using std::unordered_map;
#else
#include <boost/unordered_map.hpp>
using boost::unordered_map;
#endif
#include "commons.hpp"
#include "parameters.hpp"
#include "SIPL/Exceptions.hpp"

typedef struct TubeSegmentation {
    float *Fx, *Fy, *Fz; // The GVF vector field
    float *TDF; // The TDF response
    float *radius;
    char *centerline;
    char *segmentation;
} TubeSegmentation;

class TSFOutput {
public:
	TSFOutput(OpenCL * ocl, SIPL::int3 * size);
	bool hasSegmentation() { return deviceHasSegmentation; };
	bool hasCenterlineVoxels() { return deviceHasCenterlineVoxels; };
	bool hasTDF() { return deviceHasTDF; };
	void setTDF(cl::Image3D *);
	void setSegmentation(cl::Image3D *);
	void setCenterlineVoxels(cl::Image3D *);
	void setSize(SIPL::int3 *);
	char * getSegmentation();
	char * getCenterlineVoxels();
	float * getTDF();
	SIPL::int3 * getSize();
	~TSFOutput();
private:
	cl::Image3D * oclCenterlineVoxels;
	cl::Image3D * oclSegmentation;
	cl::Image3D * oclTDF;
	SIPL::int3 * size;
	bool hostHasSegmentation;
	bool hostHasCenterlineVoxels;
	bool hostHasTDF;
	bool deviceHasTDF;
	bool deviceHasCenterlineVoxels;
	bool deviceHasSegmentation;
	char * segmentation;
	char * centerlineVoxels;
	float * TDF;
	OpenCL * ocl;
};

cl::Image3D readDatasetAndTransfer(OpenCL, std::string, paramList, SIPL::int3 *);

TSFOutput * runCircleFittingAndRidgeTraversal(OpenCL, cl::Image3D dataset, SIPL::int3 size, paramList);

TSFOutput * runCircleFittingAndNewCenterlineAlg(OpenCL, cl::Image3D dataset, SIPL::int3 size, paramList);

TSFOutput * run(std::string filename, paramList parameters, int argc, char ** argv);

#endif
