#ifndef TUBE_SEGMENTATION
#define TUBE_SEGMENTATION

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#include "OpenCLUtilities/openCLUtilities.hpp"
#include "SIPL/Types.hpp"
#include <iostream>
#include <string>
#include <set>
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
    float *FxSmall, *FySmall, *FzSmall; // The GVF vector field
    float *TDF; // The TDF response
    float *radius;
    char *centerline;
    char *segmentation;
    float *intensity;
} TubeSegmentation;

class TSFOutput {
public:
	TSFOutput(OpenCL * ocl, SIPL::int3 * size, bool TDFis16bit = false);
	bool hasSegmentation() { return deviceHasSegmentation || hostHasSegmentation; };
	bool hasCenterlineVoxels() { return deviceHasCenterlineVoxels || hostHasCenterlineVoxels; };
	bool hasTDF() { return deviceHasTDF || hostHasTDF; };
	void setTDF(cl::Image3D *);
	void setSegmentation(cl::Image3D *);
	void setCenterlineVoxels(cl::Image3D *);
	void setTDF(float *);
	void setSegmentation(char *);
	void setCenterlineVoxels(char *);
	void setSize(SIPL::int3 *);
	char * getSegmentation();
	char * getCenterlineVoxels();
	float * getTDF();
	SIPL::int3 * getSize();
	~TSFOutput();
	SIPL::int3 getShiftVector() const;
	void setShiftVector(SIPL::int3 shiftVector);
	SIPL::float3 getSpacing() const;
	void setSpacing(SIPL::float3 spacing);

private:
	cl::Image3D* oclCenterlineVoxels;
	cl::Image3D* oclSegmentation;
	cl::Image3D* oclTDF;
	SIPL::int3* size;
	SIPL::float3 spacing;
	SIPL::int3 shiftVector;
	bool TDFis16bit;
	bool hostHasSegmentation;
	bool hostHasCenterlineVoxels;
	bool hostHasTDF;
	bool deviceHasTDF;
	bool deviceHasCenterlineVoxels;
	bool deviceHasSegmentation;
	char* segmentation;
	char* centerlineVoxels;
	float* TDF;
	OpenCL* ocl;
};

class TSFGarbageCollector {
    public:
        void addMemObject(cl::Memory * mem);
        void deleteMemObject(cl::Memory * mem);
        void deleteAllMemObjects();
        ~TSFGarbageCollector();
    private:
        std::set<cl::Memory *> memObjects;
};

/*
 * For debugging.
 */
void print(paramList parameters);

cl::Image3D readDatasetAndTransfer(OpenCL &ocl, std::string, paramList &parameters, SIPL::int3 *, TSFOutput *);

void runCircleFittingAndRidgeTraversal(OpenCL *, cl::Image3D *dataset, SIPL::int3 * size, paramList &parameters, TSFOutput *);

void runCircleFittingAndNewCenterlineAlg(OpenCL *, cl::Image3D *dataset, SIPL::int3 * size, paramList &parameters, TSFOutput *);

void runCircleFittingAndTest(OpenCL *, cl::Image3D *dataset, SIPL::int3 * size, paramList &parameters, TSFOutput *);


TSFOutput * run(std::string filename, paramList &parameters, std::string kernel_dir);

#endif
