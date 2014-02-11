#ifndef INPUT_OUTPUT_H
#define INPUT_OUTPUT_H

#include "SIPL/Types.hpp"
#include <vector>
#include "parameters.hpp"
#include "commons.hpp"
using namespace SIPL;

class TSFOutput {
public:
	TSFOutput(oul::DeviceCriteria criteria, SIPL::int3 * size, bool TDFis16bit = false);
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
	oul::Context *getContext();
private:
	oul::Context *context;
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

void writeToVtkFile(paramList &parameters, std::vector<int3> vertices, std::vector<SIPL::int2> edges);

void writeDataToDisk(TSFOutput * output, std::string storageDirectory, std::string name);

#endif
