#include "inputOutput.hpp"
#include <fstream>
#include "SIPL/Exceptions.hpp"
using namespace SIPL;
using namespace cl;

template <typename T>
void writeToRaw(T * voxels, std::string filename, int SIZE_X, int SIZE_Y, int SIZE_Z) {
    FILE * file = fopen(filename.c_str(), "wb");
    fwrite(voxels, sizeof(T), SIZE_X*SIZE_Y*SIZE_Z, file);
    fclose(file);
}

void writeDataToDisk(TSFOutput * output, std::string storageDirectory, std::string name) {
	SIPL::int3 * size = output->getSize();
	if(output->hasCenterlineVoxels()) {
		// Create MHD file
		std::ofstream file;
		std::string filename = storageDirectory + name + ".centerline.mhd";
		file.open(filename.c_str());
		file << "ObjectType = Image\n";
		file << "NDims = 3\n";
		file << "DimSize = " << output->getSize()->x << " " << output->getSize()->y << " " << output->getSize()->z << "\n";
		file << "ElementSpacing = " << output->getSpacing().x << " " << output->getSpacing().y << " " << output->getSpacing().z << "\n";
		file << "ElementType = MET_CHAR\n";
		file << "ElementDataFile = " << name << ".centerline.raw\n";
		file.close();
		writeToRaw<char>(output->getCenterlineVoxels(), storageDirectory + name + ".centerline.raw", size->x, size->y, size->z);
	}

	if(output->hasSegmentation()) {
		// Create MHD file
		std::ofstream file;
		std::string filename = storageDirectory + name + ".segmentation.mhd";
		file.open(filename.c_str());
		file << "ObjectType = Image\n";
		file << "NDims = 3\n";
		file << "DimSize = " << output->getSize()->x << " " << output->getSize()->y << " " << output->getSize()->z << "\n";
		file << "ElementSpacing = " << output->getSpacing().x << " " << output->getSpacing().y << " " << output->getSpacing().z << "\n";
		file << "ElementType = MET_CHAR\n";
		file << "ElementDataFile = " << name << ".segmentation.raw\n";
		file.close();

		writeToRaw<char>(output->getSegmentation(), storageDirectory + name + ".segmentation.raw", size->x, size->y, size->z);
	}
}

void writeToVtkFile(paramList &parameters, std::vector<int3> vertices, std::vector<SIPL::int2> edges) {
	// Write to file
	std::ofstream file;
	file.open(getParamStr(parameters, "centerline-vtk-file").c_str());
	file << "# vtk DataFile Version 3.0\nvtk output\nASCII\n";
	file << "DATASET POLYDATA\nPOINTS " << vertices.size() << " int\n";
	for(int i = 0; i < vertices.size(); i++) {
		file << vertices[i].x << " " << vertices[i].y << " " << vertices[i].z << "\n";
	}

	file << "\nLINES " << edges.size() << " " << edges.size()*3 << "\n";
	for(int i = 0; i < edges.size(); i++) {
		file << "2 " << edges[i].x << " " << edges[i].y << "\n";
	}

	file.close();
}

TSFOutput::TSFOutput(oul::DeviceCriteria criteria, SIPL::int3 * size, bool TDFis16bit) {
    oul::OpenCLManager * manager = oul::OpenCLManager::getInstance();
    //manager->setDebugMode(true);
    std::vector<oul::PlatformDevices> platformDevices = manager->getDevices(criteria);
    std::vector<cl::Device> validDevices = manager->getDevicesForBestPlatform(
                            criteria, platformDevices);

    this->context = new oul::Context(validDevices,false);//TODO:, false, getParamBool(parameters, "timing"));
	this->TDFis16bit = TDFis16bit;
    OpenCL * ocl = new OpenCL;
    ocl->context = context->getContext();
	ocl->platform = context->getPlatform();
	ocl->queue = context->getQueue(0);
	ocl->device = context->getDevice(0);
	this->ocl = ocl;
	this->size = size;
	hostHasCenterlineVoxels = false;
	hostHasSegmentation = false;
	hostHasTDF = false;
	deviceHasCenterlineVoxels = false;
	deviceHasSegmentation = false;
	deviceHasTDF = false;
}

oul::Context * TSFOutput::getContext() {
    return this->context;
}

TSFOutput::~TSFOutput() {
	if(hostHasTDF)
		delete[] TDF;
	if(hostHasSegmentation)
		delete[] segmentation;
	if(hostHasCenterlineVoxels)
		delete[] centerlineVoxels;
	if(deviceHasTDF)
		delete oclTDF;
	if(deviceHasSegmentation)
		delete oclSegmentation;
	if(deviceHasCenterlineVoxels)
		delete oclCenterlineVoxels;
	delete ocl;
	delete size;
}

void TSFOutput::setTDF(Image3D * image) {
	deviceHasTDF = true;
	oclTDF = image;
}

void TSFOutput::setTDF(float * data) {
	hostHasTDF = true;
	TDF = data;
}

void TSFOutput::setSegmentation(Image3D * image) {
	deviceHasSegmentation = true;
	oclSegmentation = image;
}

void TSFOutput::setSegmentation(char * data) {
	hostHasSegmentation = true;
	segmentation = data;
}

void TSFOutput::setCenterlineVoxels(Image3D * image) {
	deviceHasCenterlineVoxels = true;
	oclCenterlineVoxels = image;
}

void TSFOutput::setCenterlineVoxels(char * data) {
	hostHasCenterlineVoxels = true;
	centerlineVoxels = data;
}

void TSFOutput::setSize(SIPL::int3 * size) {
	this->size = size;
}

float * TSFOutput::getTDF() {
	if(hostHasTDF) {
		return TDF;
	} else if(deviceHasTDF) {
		// Transfer data from device to host
		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;
		cl::size_t<3> region;
		region[0] = size->x;
		region[1] = size->y;
		region[2] = size->z;
		int totalSize = size->x*size->y*size->z;
		TDF = new float[totalSize];
		if(TDFis16bit) {
			unsigned short * tempTDF = new unsigned short[totalSize];
			ocl->queue.enqueueReadImage(*oclTDF,CL_TRUE, origin, region, 0, 0, tempTDF);
			for(int i = 0; i < totalSize;i++) {
				TDF[i] = (float)tempTDF[i] / 65535.0f;
			}
			delete[] tempTDF;
		} else {
			ocl->queue.enqueueReadImage(*oclTDF,CL_TRUE, origin, region, 0, 0, TDF);
		}
		hostHasTDF = true;
		return TDF;
	} else {
		throw SIPL::SIPLException("Trying to fetch non existing data from TSFOutput", __LINE__, __FILE__);
	}
}

char * TSFOutput::getSegmentation() {
	if(hostHasSegmentation) {
		return segmentation;
	} else if(deviceHasSegmentation) {
		// Transfer data from device to host
		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;
		cl::size_t<3> region;
		region[0] = size->x;
		region[1] = size->y;
		region[2] = size->z;
		segmentation = new char[size->x*size->y*size->z];
		ocl->queue.enqueueReadImage(*oclSegmentation,CL_TRUE, origin, region, 0, 0, segmentation);
		hostHasSegmentation = true;
		return segmentation;
	} else {
		throw SIPL::SIPLException("Trying to fetch non existing data from TSFOutput", __LINE__, __FILE__);
	}
}

char * TSFOutput::getCenterlineVoxels() {
	if(hostHasCenterlineVoxels) {
		return centerlineVoxels;
	} else if(deviceHasCenterlineVoxels) {
		// Transfer data from device to host
		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;
		cl::size_t<3> region;
		region[0] = size->x;
		region[1] = size->y;
		region[2] = size->z;
		centerlineVoxels = new char[size->x*size->y*size->z];
		ocl->queue.enqueueReadImage(*oclCenterlineVoxels,CL_TRUE, origin, region, 0, 0, centerlineVoxels);
		hostHasCenterlineVoxels = true;
		return centerlineVoxels;
	} else {
		throw SIPL::SIPLException("Trying to fetch non existing data from TSFOutput", __LINE__, __FILE__);
	}
}

SIPL::int3 * TSFOutput::getSize() {
	return size;
}

SIPL::int3 TSFOutput::getShiftVector() const {
	return shiftVector;
}

void TSFOutput::setShiftVector(SIPL::int3 shiftVector) {
	this->shiftVector = shiftVector;
}

SIPL::float3 TSFOutput::getSpacing() const {
	return spacing;
}

void TSFOutput::setSpacing(SIPL::float3 spacing) {
	this->spacing = spacing;
}
