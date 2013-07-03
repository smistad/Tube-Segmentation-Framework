#include "tube-segmentation.hpp"
#include "gradientVectorFlow.hpp"
#include "tubeDetectionFilters.hpp"
#include "ridgeTraversalCenterlineExtraction.hpp"
#include "eigenanalysisOfHessian.hpp"
#include "globalCenterlineExtraction.hpp"
#include "parallelCenterlineExtraction.hpp"
#include "inputOutput.hpp"
#include "SIPL/Types.hpp"
//#define USE_SIPL_VISUALIZATION
#ifdef USE_SIPL_VISUALIZATION
#include "SIPL/Core.hpp"
#endif
#include <boost/iostreams/device/mapped_file.hpp>
#include <queue>
#include <stack>
#include <list>
#include <cstdio>
#include <limits>
#include <fstream>
#include "timing.hpp"
#define MAX(a,b) a > b ? a : b
//#include "tsf-config.h"

//#define TIMING

// Undefine windows crap
#ifdef WIN32
#undef min
#undef max
#else
#define __stdcall
#endif


void print(paramList parameters){
	unordered_map<std::string, BoolParameter>::iterator bIt;
	unordered_map<std::string, NumericParameter>::iterator nIt;
	unordered_map<std::string, StringParameter>::iterator sIt;

	for(bIt = parameters.bools.begin(); bIt != parameters.bools.end(); ++bIt){
		std::cout << bIt->first << " = " << bIt->second.get() << " " << bIt->second.getDescription() << " "  << bIt->second.getGroup() << std::endl;
	}

	for(nIt = parameters.numerics.begin(); nIt != parameters.numerics.end(); ++nIt){
		std::cout << nIt->first << " = " << nIt->second.get() << " " << nIt->second.getDescription() << " "  << nIt->second.getGroup() << std::endl;
	}
	for(sIt = parameters.strings.begin(); sIt != parameters.strings.end(); ++sIt){
		std::cout << sIt->first << " = " << sIt->second.get() << " " << sIt->second.getDescription() << " "  << sIt->second.getGroup() << std::endl;
	}
}

int runCounter = 0;
TSFOutput * run(std::string filename, paramList &parameters, std::string kernel_dir) {

    INIT_TIMER
    OpenCL * ocl = new OpenCL;
    cl_device_type type;
    if(parameters.strings["device"].get() == "gpu") {
    	type = CL_DEVICE_TYPE_GPU;
    } else {
    	type = CL_DEVICE_TYPE_CPU;
    }
	ocl->context = createCLContext(type);
	ocl->platform = getPlatform(type, VENDOR_ANY);

    // Select first device
    VECTOR_CLASS<cl::Device> devices = ocl->context.getInfo<CL_CONTEXT_DEVICES>();
    std::cout << "Using device: " << devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;
    ocl->device = devices[0];
    if(getParamBool(parameters, "timing")) {
        ocl->queue = cl::CommandQueue(ocl->context, devices[0], CL_QUEUE_PROFILING_ENABLE);
    } else {
        ocl->queue = cl::CommandQueue(ocl->context, devices[0]);
    }

    // Query the size of available memory
    unsigned int memorySize = devices[0].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    std::cout << "Available memory on selected device " << (double)memorySize/(1024*1024) << " MB "<< std::endl;
    std::cout << "Max alloc size: " << (float)devices[0].getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()/(1024*1024) << " MB " << std::endl;

    if(ocl->platform.getInfo<CL_PLATFORM_VENDOR>().substr(0,5) == "Apple")
        setParameter(parameters, "16bit-vectors", "false");

    // Compile and create program
    if(!getParamBool(parameters, "buffers-only") && (int)devices[0].getInfo<CL_DEVICE_EXTENSIONS>().find("cl_khr_3d_image_writes") > -1) {
    	std::string filename = kernel_dir+"/kernels.cl";
        std::string buildOptions = "";
        if(getParamBool(parameters, "16bit-vectors")) {
        	buildOptions = "-D VECTORS_16BIT";
        }
        ocl->program = buildProgramFromSource(ocl->context, filename, buildOptions);
        BoolParameter v = parameters.bools["3d_write"];
        v.set(true);
        parameters.bools["3d_write"] = v;
    } else {
        std::cout << "NOTE: Writing to 3D textures is not supported on the selected device." << std::endl;
        BoolParameter v = parameters.bools["3d_write"];
        v.set(false);
        parameters.bools["3d_write"] = v;
        std::string filename = kernel_dir+"/kernels_no_3d_write.cl";
        std::string buildOptions = "";
        if(getParamBool(parameters, "16bit-vectors")) {
        	buildOptions = "-D VECTORS_16BIT";
        	std::cout << "NOTE: Forcing the use of 16 bit buffers. This is slow, but uses half the memory." << std::endl;
        }
        ocl->program = buildProgramFromSource(ocl->context, filename, buildOptions);
    }

    if(getParamBool(parameters, "timer-total")) {
		START_TIMER
    }
    SIPL::int3 * size = new SIPL::int3();
    TSFOutput * output = new TSFOutput(ocl, size, getParamBool(parameters, "16bit-vectors"));
    try {
        // Read dataset and transfer to device
        cl::Image3D * dataset = new cl::Image3D;
        ocl->GC.addMemObject(dataset);
        *dataset = readDatasetAndTransfer(*ocl, filename, parameters, size, output);

        // Calculate maximum memory usage
        double totalSize = size->x*size->y*size->z;
        double vectorTypeSize = getParamBool(parameters, "16bit-vectors") ? sizeof(short):sizeof(float);
        double peakSize = totalSize*10.0*vectorTypeSize;
        std::cout << "NOTE: Peak memory usage with current dataset size is: " << (double)peakSize/(1024*1024) << " MB " << std::endl;
        if(peakSize > memorySize) {
            std::cout << "WARNING: There may not be enough space available on the GPU to process this volume." << std::endl;
            std::cout << "WARNING: Shrink volume with " << (double)(peakSize-memorySize)*100.0/peakSize << "% (" << (double)(peakSize-memorySize)/(1024*1024) << " MB) " << std::endl;
        }

        // Run specified method on dataset
        if(getParamStr(parameters, "centerline-method") == "ridge") {
            runCircleFittingAndRidgeTraversal(ocl, dataset, size, parameters, output);
        } else if(getParamStr(parameters, "centerline-method") == "gpu") {
            runCircleFittingAndNewCenterlineAlg(ocl, dataset, size, parameters, output);
        } else if(getParamStr(parameters, "centerline-method") == "test") {
            runCircleFittingAndTest(ocl, dataset, size, parameters, output);
        }
    } catch(cl::Error e) {
    	std::string str = "OpenCL error: " + std::string(getCLErrorString(e.err()));
        ocl->GC.deleteAllMemObjects();
        delete output;

        if(e.err() == CL_INVALID_COMMAND_QUEUE && runCounter < 2) {
            std::cout << "OpenCL error: Invalid Command Queue. Retrying..." << std::endl;
            runCounter++;
            return run(filename,parameters,kernel_dir);
        }

        throw SIPL::SIPLException(str.c_str());
    }
    ocl->queue.finish();
    if(getParamBool(parameters, "timer-total")) {
		STOP_TIMER("total")
    }
    ocl->GC.deleteAllMemObjects();
    return output;
}



template <typename T>
void writeToRaw(T * voxels, std::string filename, int SIZE_X, int SIZE_Y, int SIZE_Z) {
    FILE * file = fopen(filename.c_str(), "wb");
    fwrite(voxels, sizeof(T), SIZE_X*SIZE_Y*SIZE_Z, file);
    fclose(file);
}
template <typename T>
T * readFromRaw(std::string filename, int SIZE_X, int SIZE_Y, int SIZE_Z) {
    FILE * file = fopen(filename.c_str(), "rb");
    T * data = new T[SIZE_X*SIZE_Y*SIZE_Z];
    fread(data, sizeof(T), SIZE_X*SIZE_Y*SIZE_Z, file);
    fclose(file);
    return data;
}

using SIPL::float3;
using SIPL::int3;

using namespace cl;

template <typename T>
void __stdcall freeData(cl_mem memobj, void * user_data) {
    T * data = (T *)user_data;
    delete[] data;
}

void __stdcall notify(cl_mem memobj, void * user_data) {
    std::cout << "DELETED: " << (char *)user_data << " object was deleted" << std::endl;
}







float * createBlurMask(float sigma, int * maskSizePointer) {
    int maskSize = (int)ceil(sigma/0.5f);
    if(maskSize < 1) // cap min mask size at 3x3x3
    	maskSize = 1;
    if(maskSize > 5) // cap mask size at 11x11x11
    	maskSize = 5;
    float * mask = new float[(maskSize*2+1)*(maskSize*2+1)*(maskSize*2+1)];
    float sum = 0.0f;
    for(int a = -maskSize; a < maskSize+1; a++) {
        for(int b = -maskSize; b < maskSize+1; b++) {
            for(int c = -maskSize; c < maskSize+1; c++) {
                sum += exp(-((float)(a*a+b*b+c*c) / (2*sigma*sigma)));
                mask[a+maskSize+(b+maskSize)*(maskSize*2+1)+(c+maskSize)*(maskSize*2+1)*(maskSize*2+1)] = exp(-((float)(a*a+b*b+c*c) / (2*sigma*sigma)));

            }
        }
    }
    for(int i = 0; i < (maskSize*2+1)*(maskSize*2+1)*(maskSize*2+1); i++)
        mask[i] = mask[i] / sum;

    *maskSizePointer = maskSize;

    return mask;
}
void runCircleFittingMethod(OpenCL &ocl, Image3D * dataset, SIPL::int3 size, paramList &parameters, Image3D &vectorField, Image3D &TDF, Image3D &radiusImage) {
    // Set up parameters
    const float radiusMin = getParam(parameters, "radius-min");
    const float radiusMax = getParam(parameters, "radius-max");
    const float radiusStep = getParam(parameters, "radius-step");
    const float Fmax = getParam(parameters, "fmax");
    const int totalSize = size.x*size.y*size.z;
    const bool no3Dwrite = !getParamBool(parameters, "3d_write");
    const int vectorSign = getParamStr(parameters, "mode") == "black" ? -1 : 1;
    const float smallBlurSigma = getParam(parameters, "small-blur");
	const float largeBlurSigma = getParam(parameters,"large-blur");


    cl::size_t<3> offset;
    offset[0] = 0;
    offset[1] = 0;
    offset[2] = 0;
    cl::size_t<3> region;
    region[0] = size.x;
    region[1] = size.y;
    region[2] = size.z;

    // Create kernels
    Kernel blurVolumeWithGaussianKernel(ocl.program, "blurVolumeWithGaussian");
    Kernel createVectorFieldKernel(ocl.program, "createVectorField");
    Kernel combineKernel = Kernel(ocl.program, "combine");

    cl::Event startEvent, endEvent;
    cl_ulong start, end;
    INIT_TIMER
    void * TDFsmall;
    float * radiusSmall;
    if(radiusMin < 2.5f) {
        Image3D * blurredVolume = new Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_R, CL_FLOAT), size.x, size.y, size.z);
        ocl.GC.addMemObject(blurredVolume);
    if(smallBlurSigma > 0) {
    	int maskSize = 1;
		float * mask = createBlurMask(smallBlurSigma, &maskSize);
		Buffer blurMask = Buffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*(maskSize*2+1)*(maskSize*2+1)*(maskSize*2+1), mask);
        blurMask.setDestructorCallback((void (__stdcall *)(cl_mem,void *))(freeData<float>), (void *)mask);
    	if(no3Dwrite) {
			// Create auxillary buffer
			Buffer blurredVolumeBuffer = Buffer(
					ocl.context,
					CL_MEM_WRITE_ONLY,
					sizeof(float)*totalSize
			);

			// Run blurVolumeWithGaussian on dataset
			blurVolumeWithGaussianKernel.setArg(0, *dataset);
			blurVolumeWithGaussianKernel.setArg(1, blurredVolumeBuffer);
			blurVolumeWithGaussianKernel.setArg(2, maskSize);
			blurVolumeWithGaussianKernel.setArg(3, blurMask);

			ocl.queue.enqueueNDRangeKernel(
					blurVolumeWithGaussianKernel,
					NullRange,
					NDRange(size.x,size.y,size.z),
					NullRange
			);

			ocl.queue.enqueueCopyBufferToImage(
					blurredVolumeBuffer,
					*blurredVolume,
					0,
					offset,
					region
			);
    	} else {
			// Run blurVolumeWithGaussian on processedVolume
			blurVolumeWithGaussianKernel.setArg(0, *dataset);
			blurVolumeWithGaussianKernel.setArg(1, *blurredVolume);
			blurVolumeWithGaussianKernel.setArg(2, maskSize);
			blurVolumeWithGaussianKernel.setArg(3, blurMask);
			ocl.queue.enqueueNDRangeKernel(
					blurVolumeWithGaussianKernel,
					NullRange,
					NDRange(size.x,size.y,size.z),
					NullRange
			);
    	}
    } else {
        blurredVolume = dataset;
    }

if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&startEvent);
}
    Image3D * vectorFieldSmall;
    if(no3Dwrite) {
    	bool usingTwoBuffers = false;
    	int maxZ = size.z;
        // Create auxillary buffer
        Buffer vectorFieldBuffer, vectorFieldBuffer2;
        unsigned int maxBufferSize = ocl.device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
        if(getParamBool(parameters, "16bit-vectors")) {
			if(4*sizeof(short)*totalSize < maxBufferSize) {
				vectorFieldBuffer = Buffer(ocl.context, CL_MEM_WRITE_ONLY, 4*sizeof(short)*totalSize);
			} else {
				std::cout << "NOTE: Could not fit entire vector field into one buffer. Splitting buffer in two." << std::endl;
				// create two buffers
				unsigned int limit = (float)maxBufferSize / (4*sizeof(short));
				maxZ = floor((float)limit/(size.x*size.y));
				unsigned int splitSize = maxZ*size.x*size.y*4*sizeof(short);
				vectorFieldBuffer = Buffer(ocl.context, CL_MEM_WRITE_ONLY, splitSize);
				vectorFieldBuffer2 = Buffer(ocl.context, CL_MEM_WRITE_ONLY, 4*sizeof(short)*totalSize-splitSize);
				usingTwoBuffers = true;
			}
        } else {
			if(4*sizeof(float)*totalSize < maxBufferSize) {
				vectorFieldBuffer = Buffer(ocl.context, CL_MEM_WRITE_ONLY, 4*sizeof(float)*totalSize);
			} else {
				std::cout << "NOTE: Could not fit entire vector field into one buffer. Splitting buffer in two." << std::endl;
				// create two buffers
				unsigned int limit = (float)maxBufferSize / (4*sizeof(float));
				maxZ = floor((float)limit/(size.x*size.y));
				unsigned int splitSize = maxZ*size.x*size.y*4*sizeof(float);
				vectorFieldBuffer = Buffer(ocl.context, CL_MEM_WRITE_ONLY, splitSize);
				vectorFieldBuffer2 = Buffer(ocl.context, CL_MEM_WRITE_ONLY, 4*sizeof(float)*totalSize-splitSize);
				usingTwoBuffers = true;
    		}
        }

        // Run create vector field
        createVectorFieldKernel.setArg(0, *blurredVolume);
        createVectorFieldKernel.setArg(1, vectorFieldBuffer);
        createVectorFieldKernel.setArg(2, vectorFieldBuffer2);
        createVectorFieldKernel.setArg(3, Fmax);
        createVectorFieldKernel.setArg(4, vectorSign);
        createVectorFieldKernel.setArg(5, maxZ);


        ocl.queue.enqueueNDRangeKernel(
                createVectorFieldKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NullRange
        );

        if(smallBlurSigma > 0) {
            ocl.queue.finish();
            ocl.GC.deleteMemObject(blurredVolume);
        }

        if(getParamBool(parameters, "16bit-vectors")) {
            vectorFieldSmall = new Image3D(
                ocl.context, 
                CL_MEM_READ_ONLY,
                ImageFormat(CL_RGBA, CL_SNORM_INT16),
                size.x,size.y,size.z
            );
        } else {
            vectorFieldSmall = new Image3D(
                    ocl.context, 
                    CL_MEM_READ_ONLY,
                    ImageFormat(CL_RGBA, CL_FLOAT),
                size.x,size.y,size.z
            );
        }
        ocl.GC.addMemObject(vectorFieldSmall);
        if(usingTwoBuffers) {
        	cl::size_t<3> region2;
        	region2[0] = size.x;
        	region2[1] = size.y;
        	unsigned int limit;
			if(getParamBool(parameters, "16bit-vectors")) {
				limit = (float)maxBufferSize / (4*sizeof(short));
			} else {
				limit = (float)maxBufferSize / (4*sizeof(float));
			}
        	region2[2] = floor((float)limit/(size.x*size.y));
 			ocl.queue.enqueueCopyBufferToImage(
					vectorFieldBuffer,
					*vectorFieldSmall,
					0,
					offset,
					region2
			);
 			cl::size_t<3> offset2;
 			offset2[0] = 0;
 			offset2[1] = 0;
 			offset2[2] = region2[2];
 			cl::size_t<3> region3;
 			region3[0] = size.x;
 			region3[1] = size.y;
 			region3[2] = size.z-region2[2];
			ocl.queue.enqueueCopyBufferToImage(
					vectorFieldBuffer2,
					*vectorFieldSmall,
					0,
					offset2,
					region3
			);
        } else {
			// Copy buffer contents to image
			ocl.queue.enqueueCopyBufferToImage(
					vectorFieldBuffer,
					*vectorFieldSmall,
					0,
					offset,
					region
			);
        }

    } else {
        if(getParamBool(parameters, "32bit-vectors")) {
            std::cout << "NOTE: Using 32 bit vectors" << std::endl;
            vectorFieldSmall = new Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_RGBA, CL_FLOAT), size.x, size.y, size.z);
        } else {
            std::cout << "NOTE: Using 16 bit vectors" << std::endl;
            vectorFieldSmall = new Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_RGBA, CL_SNORM_INT16), size.x, size.y, size.z);
        }
        ocl.GC.addMemObject(vectorFieldSmall);

        // Run create vector field
        createVectorFieldKernel.setArg(0, *blurredVolume);
        createVectorFieldKernel.setArg(1, *vectorFieldSmall);
        createVectorFieldKernel.setArg(2, Fmax);
        createVectorFieldKernel.setArg(3, vectorSign);

        ocl.queue.enqueueNDRangeKernel(
                createVectorFieldKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NDRange(4,4,4)
        );

    if(smallBlurSigma > 0) {
        ocl.queue.finish();
        ocl.GC.deleteMemObject(blurredVolume);
    }
    }


if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&endEvent);
    ocl.queue.finish();
    startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    endEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &end);
    std::cout << "RUNTIME of Create vector field: " << (end-start)*1.0e-6 << " ms" << std::endl;
}
if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&startEvent);
}
    // Run circle fitting TDF kernel
    Buffer * TDFsmallBuffer;
    if(getParamBool(parameters, "16bit-vectors")) {
        TDFsmallBuffer = new Buffer(ocl.context, CL_MEM_WRITE_ONLY, sizeof(short)*totalSize);
    } else {
        TDFsmallBuffer = new Buffer(ocl.context, CL_MEM_WRITE_ONLY, sizeof(float)*totalSize);
    }
    ocl.GC.addMemObject(TDFsmallBuffer);
    Buffer * radiusSmallBuffer = new Buffer(ocl.context, CL_MEM_WRITE_ONLY, sizeof(float)*totalSize);
    ocl.GC.addMemObject(radiusSmallBuffer);
    runCircleFittingTDF(ocl,size,vectorFieldSmall,TDFsmallBuffer,radiusSmallBuffer,radiusMin,3.0f,0.5f);


    if(radiusMax < 2.5) {
    	// Stop here
    	// Copy TDFsmall to TDF and radiusSmall to radiusImage
        if(getParamBool(parameters, "16bit-vectors")) {
            TDF = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_R, CL_UNORM_INT16),
				size.x, size.y, size.z);
        } else {
            TDF = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_R, CL_FLOAT),
				size.x, size.y, size.z);
        }
		ocl.queue.enqueueCopyBufferToImage(
			*TDFsmallBuffer,
			TDF,
			0,
			offset,
			region
		);
		radiusImage = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_R, CL_FLOAT),
				size.x, size.y, size.z);
		ocl.queue.enqueueCopyBufferToImage(
			*radiusSmallBuffer,
			radiusImage,
			0,
			offset,
			region
		);
        vectorField = *vectorFieldSmall;
        ocl.queue.finish();
        ocl.GC.deleteMemObject(dataset);
		return;
    } else {
        ocl.queue.finish();
        ocl.GC.deleteMemObject(vectorFieldSmall);
    }

    // TODO: cleanup the two arrays below!!!!!!!!
	// Transfer result back to host
    if(getParamBool(parameters, "16bit-vectors")) {
        TDFsmall = new unsigned short[totalSize];
        ocl.queue.enqueueReadBuffer(*TDFsmallBuffer, CL_FALSE, 0, sizeof(short)*totalSize, (unsigned short*)TDFsmall);
    } else {
        TDFsmall = new float[totalSize];
        ocl.queue.enqueueReadBuffer(*TDFsmallBuffer, CL_FALSE, 0, sizeof(float)*totalSize, (float*)TDFsmall);
    }
    radiusSmall = new float[totalSize];
    ocl.queue.enqueueReadBuffer(*radiusSmallBuffer, CL_FALSE, 0, sizeof(float)*totalSize, radiusSmall);

    ocl.queue.finish(); // This finish statement is necessary. Incorrect combine result if not present.
    ocl.GC.deleteMemObject(TDFsmallBuffer);
    ocl.GC.deleteMemObject(radiusSmallBuffer);

    if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&endEvent);
    ocl.queue.finish();
    startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    endEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &end);
    std::cout << "RUNTIME of TDF small: " << (end-start)*1.0e-6 << " ms" << std::endl;
    }

    } // end if radiusMin < 2.5


    /* Large Airways */

if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&startEvent);
}
    Image3D * blurredVolume = new Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_R, CL_FLOAT), size.x, size.y, size.z);
    ocl.GC.addMemObject(blurredVolume);
    if(largeBlurSigma > 0) {
    	int maskSize = 1;
		float * mask = createBlurMask(largeBlurSigma, &maskSize);
	    Buffer blurMask = Buffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*(maskSize*2+1)*(maskSize*2+1)*(maskSize*2+1), mask);
        blurMask.setDestructorCallback((void (__stdcall *)(cl_mem,void *))(freeData<float>), (void *)mask);
    	if(no3Dwrite) {
			// Create auxillary buffer
			Buffer blurredVolumeBuffer = Buffer(
					ocl.context,
					CL_MEM_WRITE_ONLY,
					sizeof(float)*totalSize
			);

			// Run blurVolumeWithGaussian on dataset
			blurVolumeWithGaussianKernel.setArg(0, *dataset);
			blurVolumeWithGaussianKernel.setArg(1, blurredVolumeBuffer);
			blurVolumeWithGaussianKernel.setArg(2, maskSize);
			blurVolumeWithGaussianKernel.setArg(3, blurMask);

			ocl.queue.enqueueNDRangeKernel(
					blurVolumeWithGaussianKernel,
					NullRange,
					NDRange(size.x,size.y,size.z),
					NullRange
			);

			ocl.queue.enqueueCopyBufferToImage(
					blurredVolumeBuffer,
					*blurredVolume,
					0,
					offset,
					region
			);
    	} else {
			// Run blurVolumeWithGaussian on processedVolume
			blurVolumeWithGaussianKernel.setArg(0, *dataset);
			blurVolumeWithGaussianKernel.setArg(1, *blurredVolume);
			blurVolumeWithGaussianKernel.setArg(2, maskSize);
			blurVolumeWithGaussianKernel.setArg(3, blurMask);
			ocl.queue.enqueueNDRangeKernel(
					blurVolumeWithGaussianKernel,
					NullRange,
					NDRange(size.x,size.y,size.z),
					NullRange
			);
    	}
    } else {
        blurredVolume = dataset;
    }
    if(largeBlurSigma > 0) {
        ocl.queue.finish();
        ocl.GC.deleteMemObject(dataset);
    }


if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&endEvent);
    ocl.queue.finish();
    startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    endEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &end);
    std::cout << "RUNTIME blurring: " << (end-start)*1.0e-6 << " ms" << std::endl;
}
if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&startEvent);
}
	Image3D * initVectorField;
   if(no3Dwrite) {
		bool usingTwoBuffers = false;
    	int maxZ = size.z;
        // Create auxillary buffer
        Buffer vectorFieldBuffer, vectorFieldBuffer2;
        unsigned int maxBufferSize = ocl.device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
        if(getParamBool(parameters, "16bit-vectors")) {
			initVectorField = new Image3D(ocl.context, CL_MEM_READ_ONLY, ImageFormat(CL_RGBA, CL_SNORM_INT16), size.x, size.y, size.z);
			ocl.GC.addMemObject(initVectorField);
			if(4*sizeof(short)*totalSize < maxBufferSize) {
				vectorFieldBuffer = Buffer(ocl.context, CL_MEM_WRITE_ONLY, 4*sizeof(short)*totalSize);
			} else {
				std::cout << "NOTE: Could not fit entire vector field into one buffer. Splitting buffer in two." << std::endl;
				// create two buffers
				unsigned int limit = (float)maxBufferSize / (4*sizeof(short));
				maxZ = floor((float)limit/(size.x*size.y));
				unsigned int splitSize = maxZ*size.x*size.y*4*sizeof(short);
				vectorFieldBuffer = Buffer(ocl.context, CL_MEM_WRITE_ONLY, splitSize);
				vectorFieldBuffer2 = Buffer(ocl.context, CL_MEM_WRITE_ONLY, 4*sizeof(short)*totalSize-splitSize);
				usingTwoBuffers = true;
			}
        } else {
			initVectorField = new Image3D(ocl.context, CL_MEM_READ_ONLY, ImageFormat(CL_RGBA, CL_FLOAT), size.x, size.y, size.z);
			ocl.GC.addMemObject(initVectorField);
			if(4*sizeof(float)*totalSize < maxBufferSize) {
				vectorFieldBuffer = Buffer(ocl.context, CL_MEM_WRITE_ONLY, 4*sizeof(float)*totalSize);
			} else {
				std::cout << "NOTE: Could not fit entire vector field into one buffer. Splitting buffer in two." << std::endl;
				// create two buffers
				unsigned int limit = (float)maxBufferSize / (4*sizeof(float));
				maxZ = floor((float)limit/(size.x*size.y));
				unsigned int splitSize = maxZ*size.x*size.y*4*sizeof(float);
				vectorFieldBuffer = Buffer(ocl.context, CL_MEM_WRITE_ONLY, splitSize);
				vectorFieldBuffer2 = Buffer(ocl.context, CL_MEM_WRITE_ONLY, 4*sizeof(float)*totalSize-splitSize);
				usingTwoBuffers = true;
    		}
        }

        // Run create vector field
        createVectorFieldKernel.setArg(0, *blurredVolume);
        createVectorFieldKernel.setArg(1, vectorFieldBuffer);
        createVectorFieldKernel.setArg(2, vectorFieldBuffer2);
        createVectorFieldKernel.setArg(3, Fmax);
        createVectorFieldKernel.setArg(4, vectorSign);
        createVectorFieldKernel.setArg(5, maxZ);


        ocl.queue.enqueueNDRangeKernel(
                createVectorFieldKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NullRange
        );

        ocl.queue.finish();
        ocl.GC.deleteMemObject(blurredVolume);

        if(usingTwoBuffers) {
        	cl::size_t<3> region2;
        	region2[0] = size.x;
        	region2[1] = size.y;
        	unsigned int limit;
			if(getParamBool(parameters, "16bit-vectors")) {
				limit = (float)maxBufferSize / (4*sizeof(short));
			} else {
				limit = (float)maxBufferSize / (4*sizeof(float));
			}
        	region2[2] = floor((float)limit/(size.x*size.y));
 			ocl.queue.enqueueCopyBufferToImage(
					vectorFieldBuffer,
					*initVectorField,
					0,
					offset,
					region2
			);
 			cl::size_t<3> offset2;
 			offset2[0] = 0;
 			offset2[1] = 0;
 			offset2[2] = region2[2];
 			cl::size_t<3> region3;
 			region3[0] = size.x;
 			region3[1] = size.y;
 			region3[2] = size.z-region2[2];
			ocl.queue.enqueueCopyBufferToImage(
					vectorFieldBuffer2,
					*initVectorField,
					0,
					offset2,
					region3
			);
        } else {
			// Copy buffer contents to image
			ocl.queue.enqueueCopyBufferToImage(
					vectorFieldBuffer,
					*initVectorField,
					0,
					offset,
					region
			);
        }


    } else {
        if(getParamBool(parameters, "32bit-vectors")) {
            initVectorField = new Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_RGBA, CL_FLOAT), size.x, size.y, size.z);
        } else {
            initVectorField = new Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_RGBA, CL_SNORM_INT16), size.x, size.y, size.z);
        }
        ocl.GC.addMemObject(initVectorField);


        // Run create vector field
        createVectorFieldKernel.setArg(0, *blurredVolume);
        createVectorFieldKernel.setArg(1, *initVectorField);
        createVectorFieldKernel.setArg(2, Fmax);
        createVectorFieldKernel.setArg(3, vectorSign);

        ocl.queue.enqueueNDRangeKernel(
                createVectorFieldKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NDRange(4,4,4)
        );

        ocl.queue.finish();
        ocl.GC.deleteMemObject(blurredVolume);
    }

if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&endEvent);
    ocl.queue.finish();
    startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    endEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &end);
    std::cout << "RUNTIME Create vector field: " << (end-start)*1.0e-6 << " ms" << std::endl;
}
if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&startEvent);
}
	// Determine whether to use the slow GVF that use less memory or not
	bool useSlowGVF = false;
	if(no3Dwrite) {
        unsigned int maxBufferSize = ocl.device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
		if(getParamBool(parameters, "16bit-vectors")) {
			if(4*sizeof(short)*totalSize > maxBufferSize) {
				useSlowGVF = true;
			}
		} else {
			if(4*sizeof(float)*totalSize > maxBufferSize) {
				useSlowGVF = true;
			}
		}
	}
	vectorField = runFMGGVF(ocl,initVectorField,parameters,size);
	/*
	if(useSlowGVF) {
		vectorField = runGVF(ocl, initVectorField, parameters, size, true);
	} else {
		vectorField = runGVF(ocl, initVectorField, parameters, size, false);
	}*/

if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&endEvent);
    ocl.queue.finish();
    startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    endEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &end);
    std::cout << "RUNTIME of GVF: " << (end-start)*1.0e-6 << " ms" << std::endl;
}

if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&startEvent);
}
    // Run circle fitting TDF kernel on GVF result
    Buffer TDFlarge;
    if(getParamBool(parameters, "16bit-vectors")) {
        TDFlarge = Buffer(ocl.context, CL_MEM_WRITE_ONLY, sizeof(short)*totalSize);
    } else {
        TDFlarge = Buffer(ocl.context, CL_MEM_WRITE_ONLY, sizeof(float)*totalSize);
    }
    Buffer radiusLarge = Buffer(ocl.context, CL_MEM_WRITE_ONLY, sizeof(float)*totalSize);

    if(getParamBool(parameters,"use-spline-tdf")) {
        runSplineTDF(ocl,size,&vectorField,&TDFlarge,&radiusLarge,std::max(1.5f, radiusMin),radiusMax,radiusStep);
    } else {
        runCircleFittingTDF(ocl,size,&vectorField,&TDFlarge,&radiusLarge,std::max(2.5f, radiusMin),radiusMax,radiusStep);
    }

if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&endEvent);
    ocl.queue.finish();
    startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    endEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &end);
    std::cout << "RUNTIME of TDF large: " << (end-start)*1.0e-6 << " ms" << std::endl;
}
if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&startEvent);
}
	if(radiusMin < 2.5f) {
        Buffer TDFsmall2;
        if(getParamBool(parameters, "16bit-vectors")) {
            TDFsmall2 = Buffer(ocl.context, CL_MEM_READ_ONLY, sizeof(short)*totalSize);
            ocl.queue.enqueueWriteBuffer(TDFsmall2, CL_FALSE, 0, sizeof(short)*totalSize, (unsigned short*)TDFsmall);
        } else {
            TDFsmall2 = Buffer(ocl.context, CL_MEM_READ_ONLY, sizeof(float)*totalSize);
            ocl.queue.enqueueWriteBuffer(TDFsmall2, CL_FALSE, 0, sizeof(float)*totalSize, (float*)TDFsmall);
        }
        Buffer radiusSmall2 = Buffer(ocl.context, CL_MEM_READ_ONLY, sizeof(float)*totalSize);
        ocl.queue.enqueueWriteBuffer(radiusSmall2, CL_FALSE, 0, sizeof(float)*totalSize, radiusSmall);
		combineKernel.setArg(0, TDFsmall2);
		combineKernel.setArg(1, radiusSmall2);
		combineKernel.setArg(2, TDFlarge);
		combineKernel.setArg(3, radiusLarge);

		ocl.queue.enqueueNDRangeKernel(
				combineKernel,
				NullRange,
				NDRange(totalSize),
				NDRange(64)
		);
	}
    if(getParamBool(parameters, "16bit-vectors")) {
        TDF = Image3D(ocl.context, CL_MEM_READ_ONLY, ImageFormat(CL_R, CL_UNORM_INT16),
                size.x, size.y, size.z);
    } else {
        TDF = Image3D(ocl.context, CL_MEM_READ_ONLY, ImageFormat(CL_R, CL_FLOAT),
                size.x, size.y, size.z);
    }
    ocl.queue.enqueueCopyBufferToImage(
        TDFlarge,
        TDF,
        0,
        offset,
        region
    );
    radiusImage = Image3D(ocl.context, CL_MEM_READ_ONLY, ImageFormat(CL_R, CL_FLOAT),
            size.x, size.y, size.z);
    ocl.queue.enqueueCopyBufferToImage(
        radiusLarge,
        radiusImage,
        0,
        offset,
        region
    );

if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&endEvent);
    ocl.queue.finish();
    startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    endEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &end);
    std::cout << "RUNTIME of combine: " << (end-start)*1.0e-6 << " ms" << std::endl;
}
#ifdef USE_SIPL_VISUALIZATION
//if(getParamBool(parameters, "show-vector-field")) {
// get vector field
    SIPL::Volume<SIPL::float3> * vis = new SIPL::Volume<SIPL::float3>(size);
    SIPL::Volume<float> * magnitude = new SIPL::Volume<float>(size);
    TubeSegmentation T;
    T.Fx = new float[totalSize];
    T.Fy =new float[totalSize];
    T.Fz =new float[totalSize];
    float *tdfData = new float[totalSize];
    if((!getParamBool(parameters, "16bit-vectors"))) {
     // 32 bit vector fields
        float * Fs = new float[totalSize*4];
        ocl.queue.enqueueReadImage(vectorField, CL_TRUE, offset, region, 0, 0, Fs);
#pragma omp parallel for
        for(int i = 0; i < totalSize; i++) {

         SIPL::float3 v;
            v.x = Fs[i*4];
            v.y = Fs[i*4+1];
            v.z = Fs[i*4+2];
            T.Fx[i] = v.x;
            T.Fy[i] = v.y;
            T.Fz[i] = v.z;
            vis->set(i, v);
            magnitude->set(i, v.length());
        }
        delete[] Fs;

        ocl.queue.enqueueReadImage(TDF, CL_TRUE, offset, region, 0, 0, tdfData);
    } else {
     // 16 bit vector fields
        short * Fs = new short[totalSize*4];
        unsigned short * tempTDF = new unsigned short[totalSize];
        ocl.queue.enqueueReadImage(TDF, CL_TRUE, offset, region, 0, 0, tempTDF);
        ocl.queue.enqueueReadImage(vectorField, CL_TRUE, offset, region, 0, 0, Fs);
#pragma omp parallel for
        for(int i = 0; i < totalSize; i++) {
         SIPL::float3 v;
            v.x = MAX(-1.0f, Fs[i*4] / 32767.0f);
            v.y = MAX(-1.0f, Fs[i*4+1] / 32767.0f);;
            v.z = MAX(-1.0f, Fs[i*4+2] / 32767.0f);
            T.Fx[i] = v.x;
            T.Fy[i] = v.y;
            T.Fz[i] = v.z;
            vis->set(i, v);
            magnitude->set(i, v.length());
            tdfData[i] = tempTDF[i] / 65535.0f;
        }
        delete[] Fs;
        delete[] tempTDF;
    }
    //vis->show();
    magnitude->show(0.5, 1.0);


    SIPL::Volume<float> * radius= new SIPL::Volume<float>(size);
    float * rad = new float[totalSize];
ocl.queue.enqueueReadImage(radiusImage, CL_TRUE, offset, region, 0, 0, rad);
radius->setData(rad);
radius->show(40, 80);
    SIPL::Volume<float> * tdf = new SIPL::Volume<float>(size);
    tdf->setData(tdfData);
    tdf->show();
    // Create direction map
    SIPL::Volume<SIPL::float3> * directions = new SIPL::Volume<SIPL::float3>(size);
    for(int z = 0; z < size.z; z++) {
    for(int y = 0; y < size.y; y++) {
    for(int x = 0; x < size.x; x++) {
        int3 pos(x,y,z);
        SIPL::float3 value(0,0,0);
        if(radius->get(pos) > 0) {
            value = getTubeDirection(T,pos,size);
        }
        directions->set(pos,value);
    }}}
    delete[] T.Fx;
    delete[] T.Fy;
    delete[] T.Fz;
    directions->show();
//}

#endif
}


Image3D runSphereSegmentation(OpenCL ocl, Image3D &centerline, Image3D &radius, SIPL::int3 size, paramList parameters) {
	const bool no3Dwrite = !getParamBool(parameters, "3d_write");
	if(no3Dwrite) {
		cl::size_t<3> offset;
		offset[0] = 0;
		offset[1] = 0;
		offset[2] = 0;
		cl::size_t<3> region;
		region[0] = size.x;
		region[1] = size.y;
		region[2] = size.z;

		const int totalSize = size.x*size.y*size.z;
		Buffer segmentation = Buffer(
				ocl.context,
				CL_MEM_WRITE_ONLY,
				sizeof(char)*totalSize
		);
		Kernel initKernel = Kernel(ocl.program, "initCharBuffer");
		initKernel.setArg(0, segmentation);
		ocl.queue.enqueueNDRangeKernel(
				initKernel,
				NullRange,
				NDRange(totalSize),
				NDRange(4*4*4)
		);

		Kernel kernel = Kernel(ocl.program, "sphereSegmentation");
		kernel.setArg(0, centerline);
		kernel.setArg(1, radius);
		kernel.setArg(2, segmentation);
		ocl.queue.enqueueNDRangeKernel(
				kernel,
				NullRange,
			NDRange(size.x, size.y, size.z),
			NDRange(4,4,4)
		);

		Image3D segmentationImage = Image3D(
				ocl.context,
				CL_MEM_WRITE_ONLY,
				ImageFormat(CL_R, CL_UNSIGNED_INT8),
				size.x, size.y, size.z
		);

		ocl.queue.enqueueCopyBufferToImage(
				segmentation,
				segmentationImage,
				0,
				offset,
				region
		);

		return segmentationImage;
	} else {
		Image3D segmentation = Image3D(
				ocl.context,
				CL_MEM_WRITE_ONLY,
				ImageFormat(CL_R, CL_UNSIGNED_INT8),
				size.x, size.y, size.z
		);
		Kernel initKernel = Kernel(ocl.program, "init3DImage");
		initKernel.setArg(0, segmentation);
		ocl.queue.enqueueNDRangeKernel(
				initKernel,
				NullRange,
				NDRange(size.x, size.y, size.z),
				NDRange(4,4,4)
		);

		Kernel kernel = Kernel(ocl.program, "sphereSegmentation");
		kernel.setArg(0, centerline);
		kernel.setArg(1, radius);
		kernel.setArg(2, segmentation);
		ocl.queue.enqueueNDRangeKernel(
				kernel,
				NullRange,
			NDRange(size.x, size.y, size.z),
			NDRange(4,4,4)
		);

		return segmentation;
	}

}

Image3D runInverseGradientSegmentation(OpenCL &ocl, Image3D &centerline, Image3D &vectorField, Image3D &radius, SIPL::int3 size, paramList parameters) {
    const int totalSize = size.x*size.y*size.z;
	const bool no3Dwrite = !getParamBool(parameters, "3d_write");
    cl::Event startEvent, endEvent;
    cl_ulong start, end;
    if(getParamBool(parameters, "timing")) {
        ocl.queue.enqueueMarker(&startEvent);
    }

    Kernel dilateKernel = Kernel(ocl.program, "dilate");
    Kernel erodeKernel = Kernel(ocl.program, "erode");
    Kernel initGrowKernel = Kernel(ocl.program, "initGrowing");
    Kernel growKernel = Kernel(ocl.program, "grow");

    cl::size_t<3> offset;
    offset[0] = 0;
    offset[1] = 0;
    offset[2] = 0;
    cl::size_t<3> region;
    region[0] = size.x;
    region[1] = size.y;
    region[2] = size.z;


	Image3D volume = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_R, CL_SIGNED_INT8), size.x, size.y, size.z);
	ocl.queue.enqueueCopyImage(centerline, volume, offset, offset, region);

    int stopGrowing = 0;
    Buffer stop = Buffer(ocl.context, CL_MEM_WRITE_ONLY, sizeof(int));
    ocl.queue.enqueueWriteBuffer(stop, CL_FALSE, 0, sizeof(int), &stopGrowing);

    growKernel.setArg(1, vectorField);
    growKernel.setArg(3, stop);

    int i = 0;
    int minimumIterations = 0;
    if(no3Dwrite) {
        Buffer volume2 = Buffer(
                ocl.context,
                CL_MEM_READ_WRITE,
                sizeof(char)*totalSize
        );
        ocl.queue.enqueueCopyImageToBuffer(
                volume,
                volume2,
                offset,
                region,
                0
        );
        initGrowKernel.setArg(0, volume);
        initGrowKernel.setArg(1, volume2);
        initGrowKernel.setArg(2, radius);
        ocl.queue.enqueueNDRangeKernel(
            initGrowKernel,
            NullRange,
            NDRange(size.x, size.y, size.z),
            NullRange
        );
        ocl.queue.enqueueCopyBufferToImage(
                volume2,
                volume,
                0,
                offset,
                region
        );
        growKernel.setArg(0, volume);
        growKernel.setArg(2, volume2);
        while(stopGrowing == 0) {
            if(i > minimumIterations) {
                stopGrowing = 1;
                ocl.queue.enqueueWriteBuffer(stop, CL_TRUE, 0, sizeof(int), &stopGrowing);
            }

            ocl.queue.enqueueNDRangeKernel(
                    growKernel,
                    NullRange,
                        NDRange(size.x, size.y, size.z),
                        NullRange
                    );
            if(i > minimumIterations)
                ocl.queue.enqueueReadBuffer(stop, CL_TRUE, 0, sizeof(int), &stopGrowing);
            i++;
            ocl.queue.enqueueCopyBufferToImage(
                    volume2,
                    volume,
                    0,
                    offset,
                    region
            );
        }

    } else {
        Image3D volume2 = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_R, CL_SIGNED_INT8), size.x, size.y, size.z);
        ocl.queue.enqueueCopyImage(volume, volume2, offset, offset, region);
        initGrowKernel.setArg(0, volume);
        initGrowKernel.setArg(1, volume2);
        initGrowKernel.setArg(2, radius);
        ocl.queue.enqueueNDRangeKernel(
            initGrowKernel,
            NullRange,
            NDRange(size.x, size.y, size.z),
            NDRange(4,4,4)
        );
        while(stopGrowing == 0) {
            if(i > minimumIterations) {
                stopGrowing = 1;
                ocl.queue.enqueueWriteBuffer(stop, CL_FALSE, 0, sizeof(int), &stopGrowing);
            }
            if(i % 2 == 0) {
                growKernel.setArg(0, volume);
                growKernel.setArg(2, volume2);
            } else {
                growKernel.setArg(0, volume2);
                growKernel.setArg(2, volume);
            }

            ocl.queue.enqueueNDRangeKernel(
                    growKernel,
                    NullRange,
                    NDRange(size.x, size.y, size.z),
                    NDRange(4,4,4)
                    );
            if(i > minimumIterations)
                ocl.queue.enqueueReadBuffer(stop, CL_TRUE, 0, sizeof(int), &stopGrowing);
            i++;
        }

    }

    std::cout << "segmentation result grown in " << i << " iterations" << std::endl;

    if(no3Dwrite) {
        Buffer volumeBuffer = Buffer(
                ocl.context,
                CL_MEM_WRITE_ONLY,
                sizeof(char)*totalSize
        );
        dilateKernel.setArg(0, volume);
        dilateKernel.setArg(1, volumeBuffer);

        ocl.queue.enqueueNDRangeKernel(
            dilateKernel,
            NullRange,
            NDRange(size.x, size.y, size.z),
            NullRange
        );

        ocl.queue.enqueueCopyBufferToImage(
                volumeBuffer,
                volume,
                0,
                offset,
                region);

        erodeKernel.setArg(0, volume);
        erodeKernel.setArg(1, volumeBuffer);

        ocl.queue.enqueueNDRangeKernel(
            erodeKernel,
            NullRange,
            NDRange(size.x, size.y, size.z),
            NullRange
        );
        ocl.queue.enqueueCopyBufferToImage(
            volumeBuffer,
            volume,
            0,
            offset,
            region
        );
    } else {
        Image3D volume2 = Image3D(
                ocl.context,
                CL_MEM_READ_WRITE,
                ImageFormat(CL_R, CL_SIGNED_INT8),
                size.x, size.y, size.z
        );

        Kernel init3DImage(ocl.program, "init3DImage");
        init3DImage.setArg(0, volume2);
        ocl.queue.enqueueNDRangeKernel(
            init3DImage,
            NullRange,
            NDRange(size.x, size.y, size.z),
            NullRange
        );

        dilateKernel.setArg(0, volume);
        dilateKernel.setArg(1, volume2);

        ocl.queue.enqueueNDRangeKernel(
            dilateKernel,
            NullRange,
            NDRange(size.x, size.y, size.z),
            NullRange
        );

        erodeKernel.setArg(0, volume2);
        erodeKernel.setArg(1, volume);

        ocl.queue.enqueueNDRangeKernel(
            erodeKernel,
            NullRange,
            NDRange(size.x, size.y, size.z),
            NullRange
        );
    }
if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&endEvent);
    ocl.queue.finish();
    startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    endEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &end);
    std::cout << "RUNTIME of segmentation: " << (end-start)*1.0e-6 << " ms" << std::endl;
}

    return volume;
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

void runCircleFittingAndNewCenterlineAlg(OpenCL * ocl, cl::Image3D * dataset, SIPL::int3 * size, paramList &parameters, TSFOutput * output) {
    INIT_TIMER
    Image3D vectorField, radius;
    Image3D * TDF = new Image3D;
    const int totalSize = size->x*size->y*size->z;
	const bool no3Dwrite = !getParamBool(parameters, "3d_write");

    cl::size_t<3> offset;
    offset[0] = 0;
    offset[1] = 0;
    offset[2] = 0;
    cl::size_t<3> region;
    region[0] = size->x;
    region[1] = size->y;
    region[2] = size->z;

    runCircleFittingMethod(*ocl, dataset, *size, parameters, vectorField, *TDF, radius);
    output->setTDF(TDF);
    if(getParamBool(parameters, "tdf-only"))
    	return;

    Image3D * centerline = new Image3D;
    *centerline = runNewCenterlineAlg(*ocl, *size, parameters, vectorField, *TDF, radius);
    output->setCenterlineVoxels(centerline);

    Image3D * segmentation = new Image3D;
    if(!getParamBool(parameters, "no-segmentation")) {
    	if(!getParamBool(parameters, "sphere-segmentation")) {
			*segmentation = runInverseGradientSegmentation(*ocl, *centerline, vectorField, radius, *size, parameters);
    	} else {
			*segmentation = runSphereSegmentation(*ocl, *centerline, radius, *size, parameters);
    	}
    	output->setSegmentation(segmentation);
    }

	if(getParamStr(parameters, "storage-dir") != "off") {
		writeDataToDisk(output, getParamStr(parameters, "storage-dir"), getParamStr(parameters, "storage-name"));
    }

}

#ifdef USE_SIPL_VISUALIZATION
SIPL::Volume<float3> * visualizeSegments(std::vector<Segment *> segments, int3 size) {
	SIPL::Volume<float3> * connections = new SIPL::Volume<float3>(size);
    for(Segment * s : segments) {
    	for(int i = 0; i < s->sections.size()-1; i++) {
    		CrossSection * a = s->sections[i];
    		CrossSection * b = s->sections[i+1];
			int distance = ceil(a->pos.distance(b->pos));
			float3 direction(b->pos.x-a->pos.x,b->pos.y-a->pos.y,b->pos.z-a->pos.z);
			for(int i = 0; i < distance; i++) {
				float frac = (float)i/distance;
				float3 n = a->pos + frac*direction;
				int3 in(round(n.x),round(n.y),round(n.z));
				float3 v = connections->get(in);
				v.x = 1.0f;
				connections->set(in, v);
			}
		}
		for(Connection * c : s->connections) {
			CrossSection * a = c->source_section;
			CrossSection * b = c->target_section;
			int distance = ceil(a->pos.distance(b->pos));
			float3 direction(b->pos.x-a->pos.x,b->pos.y-a->pos.y,b->pos.z-a->pos.z);
			for(int i = 0; i < distance; i++) {
				float frac = (float)i/distance;
				float3 n = a->pos + frac*direction;
				int3 in(round(n.x),round(n.y),round(n.z));
				float3 v = connections->get(in);
				v.y = 1.0f;
				connections->set(in, v);
			}

		}
    }
    connections->showMIP();
    return connections;
}
#endif

void runCircleFittingAndTest(OpenCL * ocl, cl::Image3D * dataset, SIPL::int3 * size, paramList &parameters, TSFOutput * output) {
    INIT_TIMER
    Image3D vectorField, radius, vectorFieldSmall;
    Image3D * TDF = new Image3D;
    const int totalSize = size->x*size->y*size->z;
	const bool no3Dwrite = !getParamBool(parameters, "3d_write");

    cl::size_t<3> offset;
    offset[0] = 0;
    offset[1] = 0;
    offset[2] = 0;
    cl::size_t<3> region;
    region[0] = size->x;
    region[1] = size->y;
    region[2] = size->z;

    runCircleFittingMethod(*ocl, dataset, *size, parameters, vectorField, *TDF, radius);


    // Transfer from device to host
    TubeSegmentation TS;
    TS.Fx = new float[totalSize];
    TS.Fy = new float[totalSize];
    TS.Fz = new float[totalSize];
    TS.FxSmall = new float[totalSize];
    TS.FySmall = new float[totalSize];
    TS.FzSmall = new float[totalSize];
    if((no3Dwrite && !getParamBool(parameters, "16bit-vectors")) || getParamBool(parameters, "32bit-vectors")) {
    	// 32 bit vector fields
        float * Fs = new float[totalSize*4];
        ocl->queue.enqueueReadImage(vectorField, CL_TRUE, offset, region, 0, 0, Fs);
#pragma omp parallel for
        for(int i = 0; i < totalSize; i++) {
            TS.Fx[i] = Fs[i*4];
            TS.Fy[i] = Fs[i*4+1];
            TS.Fz[i] = Fs[i*4+2];
        }
        delete[] Fs;
        if(getParam(parameters, "radius-min") < 2.5) {
		float * FsSmall = new float[totalSize*4];
        ocl->queue.enqueueReadImage(vectorFieldSmall, CL_TRUE, offset, region, 0, 0, FsSmall);
#pragma omp parallel for
        for(int i = 0; i < totalSize; i++) {
            TS.FxSmall[i] = FsSmall[i*4];
            TS.FySmall[i] = FsSmall[i*4+1];
            TS.FzSmall[i] = FsSmall[i*4+2];
        }
        delete[] FsSmall;
        }

    } else {
    	// 16 bit vector fields
        short * Fs = new short[totalSize*4];
        ocl->queue.enqueueReadImage(vectorField, CL_TRUE, offset, region, 0, 0, Fs);
#pragma omp parallel for
        for(int i = 0; i < totalSize; i++) {
            TS.Fx[i] = MAX(-1.0f, Fs[i*4] / 32767.0f);
            TS.Fy[i] = MAX(-1.0f, Fs[i*4+1] / 32767.0f);;
            TS.Fz[i] = MAX(-1.0f, Fs[i*4+2] / 32767.0f);
        }
        delete[] Fs;
        if(getParam(parameters, "radius-min") < 2.5) {
		short * FsSmall = new short[totalSize*4];
        ocl->queue.enqueueReadImage(vectorFieldSmall, CL_TRUE, offset, region, 0, 0, FsSmall);
#pragma omp parallel for
        for(int i = 0; i < totalSize; i++) {
            TS.FxSmall[i] = MAX(-1.0f, FsSmall[i*4] / 32767.0f);
            TS.FySmall[i] = MAX(-1.0f, FsSmall[i*4+1] / 32767.0f);
            TS.FzSmall[i] = MAX(-1.0f, FsSmall[i*4+2] / 32767.0f);
        }
        delete[] FsSmall;
        }

    }
    TS.radius = new float[totalSize];
    TS.TDF = new float[totalSize];
    //TS.intensity = new float[totalSize];
    ocl->queue.enqueueReadImage(*TDF, CL_TRUE, offset, region, 0, 0, TS.TDF);
    output->setTDF(TS.TDF);
    ocl->queue.enqueueReadImage(radius, CL_TRUE, offset, region, 0, 0, TS.radius);
    //ocl->queue.enqueueReadImage(dataset, CL_TRUE, offset, region, 0, 0, TS.intensity);

    // Create pairs of voxels with high TDF
    std::vector<CrossSection *> crossSections = createGraph(TS, *size);

    // Display pairs
	#ifdef USE_SIPL_VISUALIZATION
    SIPL::Volume<bool> * pairs = new SIPL::Volume<bool>(*size);
    pairs->fill(false);
    for(CrossSection * c : crossSections) {
    	pairs->set(c->pos, true);
    }
    pairs->showMIP();
	#endif

    // Create segments from pairs
    std::vector<Segment *> segments = createSegments(*ocl, TS, crossSections, *size);

	#ifdef USE_SIPL_VISUALIZATION
    visualizeSegments(segments, *size);
	#endif

    // Create connections between segments
    std::cout << "creating connections..." << std::endl;
    std::cout << "number of segments is " << segments.size() << std::endl;
    createConnections(TS, segments, *size);
    std::cout << "finished creating connections." << std::endl;
    std::cout << "number of segments is " << segments.size() << std::endl;

    // Display connections, in a separate color for instance
	#ifdef USE_SIPL_VISUALIZATION
    visualizeSegments(segments, *size);
	#endif

    // Do minimum spanning tree on segments, where each segment is a node and the connetions are edges
    // must also select a root segment
    std::cout << "running minimum spanning tree" << std::endl;
    int root = selectRoot(segments);
    segments = minimumSpanningTree(segments[root], *size);
    std::cout << "finished running minimum spanning tree" << std::endl;
    std::cout << "number of segments is " << segments.size() << std::endl;

    // Visualize
	#ifdef USE_SIPL_VISUALIZATION
    visualizeSegments(segments, *size);
	#endif

    // Display which connections have been retained and which are removed

    // create depth first ordering
    std::cout << "creating depth first ordering..." << std::endl;
    int Ns;
    int * depthFirstOrderingOfSegments = createDepthFirstOrdering(segments, root, Ns);
    std::cout << "finished creating depth first ordering" << std::endl;
    std::cout << "Ns is " << Ns << std::endl;
    std::cout << "root is " << root << std::endl;

	// have to take into account that not all segments are part of the final tree, for instance, return Ns
    // Do the dynamic programming algorithm for locating the best subtree
    std::cout << "finding optimal subtree..." << std::endl;
    std::vector<Segment *> finalSegments = findOptimalSubtree(segments, depthFirstOrderingOfSegments, Ns);
    std::cout << "finished." << std::endl;
    std::cout << "number of segments is " << finalSegments.size() << std::endl;

    // TODO Display final segments and the connections
	#ifdef USE_SIPL_VISUALIZATION
    visualizeSegments(finalSegments, *size);
	#endif

    char * centerline = new char[totalSize]();
    std::vector<int3> vertices;
    std::vector<SIPL::int2> edges;
    int counter = 0;
    for(int j = 0; j < finalSegments.size(); j++) {
    	Segment * s = finalSegments[j];
    	for(int i = 0; i < s->sections.size()-1; i++) {
    		CrossSection * a = s->sections[i];
    		CrossSection * b = s->sections[i+1];
    		vertices.push_back(a->pos);
    		vertices.push_back(b->pos);
    		// TODO: NB there are some cases in which a == b here. FIXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    		//std::cout << a->index << " " << b->index << std::endl;
    		a->index = counter;
    		b->index = counter+1;
    		//std::cout << a->index << " " << b->index << std::endl;
    		counter += 2;
    		edges.push_back(SIPL::int2(a->index, b->index));
			int distance = ceil(a->pos.distance(b->pos));
			float3 direction(b->pos.x-a->pos.x,b->pos.y-a->pos.y,b->pos.z-a->pos.z);
			for(int i = 0; i < distance; i++) {
				float frac = (float)i/distance;
				float3 n = a->pos + frac*direction;
				int3 in(round(n.x),round(n.y),round(n.z));
				centerline[in.x+in.y*size->x+in.z*size->x*size->y] = 1;
			}
		}
    	for(int i = 0; i < s->connections.size(); i++) {
    		Connection * c = s->connections[i];
			CrossSection * a = c->source_section;
			CrossSection * b = c->target_section;
    		vertices.push_back(a->pos);
    		vertices.push_back(b->pos);
    		a->index = counter;
    		b->index = counter+1;
    		counter += 2;
    		edges.push_back(SIPL::int2(a->index, b->index));
			int distance = ceil(a->pos.distance(b->pos));
			float3 direction(b->pos.x-a->pos.x,b->pos.y-a->pos.y,b->pos.z-a->pos.z);
			for(int i = 0; i < distance; i++) {
				float frac = (float)i/distance;
				float3 n = a->pos + frac*direction;
				int3 in(round(n.x),round(n.y),round(n.z));
				centerline[in.x+in.y*size->x+in.z*size->x*size->y] = 1;
			}

		}
    }
    output->setCenterlineVoxels(centerline);
    if(getParamStr(parameters, "centerline-vtk-file") != "off") {
    	writeToVtkFile(parameters, vertices, edges);
    }


    Image3D * volume = new Image3D;
    if(!getParamBool(parameters, "no-segmentation")) {
        *volume = Image3D(ocl->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, ImageFormat(CL_R, CL_SIGNED_INT8), size->x, size->y, size->z, 0, 0, centerline);
		if(!getParamBool(parameters, "sphere-segmentation")) {
			*volume = runInverseGradientSegmentation(*ocl, *volume, vectorField, radius, *size, parameters);
    	} else {
			*volume = runSphereSegmentation(*ocl,*volume, radius, *size, parameters);
    	}
		output->setSegmentation(volume);
    }



	if(getParamStr(parameters, "storage-dir") != "off") {
        writeDataToDisk(output, getParamStr(parameters, "storage-dir"), getParamStr(parameters, "storage-name"));
    }

}


void runCircleFittingAndRidgeTraversal(OpenCL * ocl, Image3D * dataset, SIPL::int3 * size, paramList &parameters, TSFOutput * output) {
    
    INIT_TIMER
    cl::Event startEvent, endEvent;
    cl_ulong start, end;
    Image3D vectorField, radius,vectorFieldSmall;
    Image3D * TDF = new Image3D;
    TubeSegmentation TS;
    runCircleFittingMethod(*ocl, dataset, *size, parameters, vectorField, *TDF, radius);
    output->setTDF(TDF);
    const int totalSize = size->x*size->y*size->z;
	const bool no3Dwrite = !getParamBool(parameters, "3d_write");

    cl::size_t<3> offset;
    offset[0] = 0;
    offset[1] = 0;
    offset[2] = 0;
    cl::size_t<3> region;
    region[0] = size->x;
    region[1] = size->y;
    region[2] = size->z;

    START_TIMER
    // Transfer buffer back to host
    TS.Fx = new float[totalSize];
    TS.Fy = new float[totalSize];
    TS.Fz = new float[totalSize];
    TS.TDF = new float[totalSize];
    if(!getParamBool(parameters, "16bit-vectors")) {
    	// 32 bit vector fields
        float * Fs = new float[totalSize*4];
        ocl->queue.enqueueReadImage(vectorField, CL_TRUE, offset, region, 0, 0, Fs);
#pragma omp parallel for
        for(int i = 0; i < totalSize; i++) {
            TS.Fx[i] = Fs[i*4];
            TS.Fy[i] = Fs[i*4+1];
            TS.Fz[i] = Fs[i*4+2];
        }
        delete[] Fs;
        ocl->queue.enqueueReadImage(*TDF, CL_TRUE, offset, region, 0, 0, TS.TDF);
    } else {
    	// 16 bit vector fields
        short * Fs = new short[totalSize*4];
        ocl->queue.enqueueReadImage(vectorField, CL_TRUE, offset, region, 0, 0, Fs);
#pragma omp parallel for
        for(int i = 0; i < totalSize; i++) {
            TS.Fx[i] = MAX(-1.0f, Fs[i*4] / 32767.0f);
            TS.Fy[i] = MAX(-1.0f, Fs[i*4+1] / 32767.0f);;
            TS.Fz[i] = MAX(-1.0f, Fs[i*4+2] / 32767.0f);
        }
        delete[] Fs;

        // Convert 16 bit TDF to 32 bit
        unsigned short * tempTDF = new unsigned short[totalSize];
        ocl->queue.enqueueReadImage(*TDF, CL_TRUE, offset, region, 0, 0, tempTDF);
#pragma omp parallel for
        for(int i = 0; i < totalSize; i++) {
            TS.TDF[i] = (float)tempTDF[i] / 65535.0f;
        }
        delete[] tempTDF;
    }
    TS.radius = new float[totalSize];
    output->setTDF(TS.TDF);
    ocl->queue.enqueueReadImage(radius, CL_TRUE, offset, region, 0, 0, TS.radius);
    std::stack<CenterlinePoint> centerlineStack;
    TS.centerline = runRidgeTraversal(TS, *size, parameters, centerlineStack);
    output->setCenterlineVoxels(TS.centerline);

    if(getParamBool(parameters, "timing")) {
        ocl->queue.finish();
        STOP_TIMER("Centerline extraction + transfer of data back and forth")
        ocl->queue.enqueueMarker(&startEvent);
    }

    Image3D * volume = new Image3D;
    if(!getParamBool(parameters, "no-segmentation")) {
        *volume = Image3D(ocl->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, ImageFormat(CL_R, CL_SIGNED_INT8), size->x, size->y, size->z, 0, 0, TS.centerline);
		if(!getParamBool(parameters, "sphere-segmentation")) {
			*volume = runInverseGradientSegmentation(*ocl, *volume, vectorField, radius, *size, parameters);
    	} else {
			*volume = runSphereSegmentation(*ocl,*volume, radius, *size, parameters);
    	}
		output->setSegmentation(volume);
    }


    if(getParamStr(parameters, "storage-dir") != "off") {
        writeDataToDisk(output, getParamStr(parameters, "storage-dir"), getParamStr(parameters, "storage-name"));
    }

}


void __stdcall unmapRawfile(cl_mem memobj, void * user_data) {
    boost::iostreams::mapped_file_source * file = (boost::iostreams::mapped_file_source *)user_data;
    file->close();
    delete[] file;
}

template <class T> 
float getMaximum(void * data, const int totalSize) {
    T * newDataPointer = (T *)data;
    T maximum = std::numeric_limits<T>::min();
    for(int i = 0; i < totalSize; i++) 
        maximum = std::max(maximum, newDataPointer[i]);

    return (float)maximum;
}

template <class T> 
float getMinimum(void * data, const int totalSize) {
    T * newDataPointer = (T *)data;
    T minimum = std::numeric_limits<T>::max();
    for(int i = 0; i < totalSize; i++) 
        minimum = std::min(minimum, newDataPointer[i]);

    return (float)minimum;
}

template <typename T>
void getLimits(paramList parameters, void * data, const int totalSize, float * minimum, float * maximum) {
    if(getParamStr(parameters, "minimum") != "off") {
        *minimum = atof(getParamStr(parameters, "minimum").c_str());
    } else {
        std::cout << "NOTE: minimum parameter not set, finding minimum automatically." << std::endl;
        *minimum = getMinimum<T>(data, totalSize);
        std::cout << "NOTE: minimum found to be " << *minimum << std::endl;
    }
            
    if(getParamStr(parameters, "maximum") != "off") {
        *maximum = atof(getParamStr(parameters, "maximum").c_str());
    } else {
        std::cout << "NOTE: maximum parameter not set, finding maximum automatically." << std::endl;
        *maximum = getMaximum<T>(data, totalSize);
        std::cout << "NOTE: maximum found to be " << *maximum << std::endl;
    }
}

boost::iostreams::mapped_file_source * file;
Image3D readDatasetAndTransfer(OpenCL &ocl, std::string filename, paramList &parameters, SIPL::int3 * size, TSFOutput * output) {
    cl_ulong start, end;
    Event startEvent, endEvent;
    if(getParamBool(parameters, "timing")) {
        ocl.queue.enqueueMarker(&startEvent);
    }
    INIT_TIMER
    START_TIMER
    // Read mhd file, determine file type
    std::fstream mhdFile;
    mhdFile.open(filename.c_str(), std::fstream::in);
    if(!mhdFile) {
    	throw SIPL::IOException(filename.c_str(), __LINE__, __FILE__);
    }
    std::string typeName = "";
    std::string rawFilename = "";
    bool typeFound = false, sizeFound = false, rawFilenameFound = false;
    SIPL::float3 spacing(1,1,1);
    do {
        std::string line;
        std::getline(mhdFile, line);
        if(line.substr(0, 11) == "ElementType") {
            typeName = line.substr(11+3);
            typeFound = true;
        } else if(line.substr(0, 15) == "ElementDataFile") {
            rawFilename = line.substr(15+3);
            rawFilenameFound = true;

            // Remove any trailing spaces
            int pos = rawFilename.find(" ");
            if(pos > 0)
            rawFilename = rawFilename.substr(0,pos);
            
            // Get path name
            pos = filename.rfind('/');
            if(pos > 0)
                rawFilename = filename.substr(0,pos+1) + rawFilename;
        } else if(line.substr(0, 7) == "DimSize") {
            std::string sizeString = line.substr(7+3);
            std::string sizeX = sizeString.substr(0,sizeString.find(" "));
            sizeString = sizeString.substr(sizeString.find(" ")+1);
            std::string sizeY = sizeString.substr(0,sizeString.find(" "));
            sizeString = sizeString.substr(sizeString.find(" ")+1);
            std::string sizeZ = sizeString.substr(0,sizeString.find(" "));

            size->x = atoi(sizeX.c_str());
            size->y = atoi(sizeY.c_str());
            size->z = atoi(sizeZ.c_str());

            sizeFound = true;
		} else if(line.substr(0, 14) == "ElementSpacing") {
            std::string sizeString = line.substr(14+3);
            std::string sizeX = sizeString.substr(0,sizeString.find(" "));
            sizeString = sizeString.substr(sizeString.find(" ")+1);
            std::string sizeY = sizeString.substr(0,sizeString.find(" "));
            sizeString = sizeString.substr(sizeString.find(" ")+1);
            std::string sizeZ = sizeString.substr(0,sizeString.find(" "));

            spacing.x = atof(sizeX.c_str());
            spacing.y = atof(sizeY.c_str());
            spacing.z = atof(sizeZ.c_str());
        }

    } while(!mhdFile.eof());

    // Remove any trailing spaces
    int pos = typeName.find(" ");
    if(pos > 0)
        typeName = typeName.substr(0,pos);

    if(!typeFound || !sizeFound || !rawFilenameFound) {
        throw SIPL::SIPLException("Error reading mhd file. Type, filename or size not found", __LINE__, __FILE__);
    }

    // Read dataset by memory mapping the file and transfer to device
    Image3D dataset;
    int type = 0;
    void * data;
    file = new boost::iostreams::mapped_file_source[1];
    cl::size_t<3> offset;
    offset[0] = 0;
    offset[1] = 0;
    offset[2] = 0;
    cl::size_t<3> region2;
    region2[0] = size->x;
    region2[1] = size->y;
    region2[2] = size->z;
    float minimum = 0.0f, maximum = 1.0f;
    const int totalSize = size->x*size->y*size->z;
    ImageFormat imageFormat;

    if(typeName == "MET_SHORT") {
        type = 1;
        file->open(rawFilename, size->x*size->y*size->z*sizeof(short));
        imageFormat = ImageFormat(CL_R, CL_SIGNED_INT16);
        dataset = Image3D(
                ocl.context, 
                CL_MEM_READ_ONLY,
                imageFormat,
                size->x, size->y, size->z
        );
        data = (void *)file->data();
        ocl.queue.enqueueWriteImage(dataset, CL_FALSE, offset, region2, 0, 0, data);
        getLimits<short>(parameters, data, totalSize, &minimum, &maximum);
    } else if(typeName == "MET_USHORT") {
        type = 2;
        file->open(rawFilename, size->x*size->y*size->z*sizeof(short));
        imageFormat = ImageFormat(CL_R, CL_UNSIGNED_INT16);
        dataset = Image3D(
                ocl.context, 
                CL_MEM_READ_ONLY,
                ImageFormat(CL_R, CL_UNSIGNED_INT16),
                size->x, size->y, size->z
        );
        data = (void *)file->data();
        ocl.queue.enqueueWriteImage(dataset, CL_FALSE, offset, region2, 0, 0, data);
        getLimits<unsigned short>(parameters, data, totalSize, &minimum, &maximum);

        if(getParamStr(parameters, "parameters") == "Lung-Airways-CT") {
        	// If parameter preset is airway and the volume loaded is unsigned;
        	// Change min and max to be unsigned as well, and change Threshold in cropping
			char * str = new char[255];
        	minimum = atof(parameters.strings["minimum"].get().c_str())+1024.0f;
        	sprintf(str, "%f", minimum);
        	parameters.strings["minimum"].set(str);
			maximum = atof(parameters.strings["maximum"].get().c_str())+1024.0f;
        	sprintf(str, "%f", maximum);
        	parameters.strings["maximum"].set(str);
        }

    } else if(typeName == "MET_CHAR") {
        type = 1;
        file->open(rawFilename, size->x*size->y*size->z*sizeof(char));
        imageFormat = ImageFormat(CL_R, CL_SIGNED_INT8);
        dataset = Image3D(
                ocl.context, 
                CL_MEM_READ_ONLY,
                imageFormat,
                size->x, size->y, size->z
        );
        data = (void *)file->data();
        ocl.queue.enqueueWriteImage(dataset, CL_FALSE, offset, region2, 0, 0, data);
        getLimits<char>(parameters, data, totalSize, &minimum, &maximum);
    } else if(typeName == "MET_UCHAR") {
        type = 2;
        file->open(rawFilename, size->x*size->y*size->z*sizeof(char));
        imageFormat = ImageFormat(CL_R, CL_UNSIGNED_INT8);
        dataset = Image3D(
                ocl.context, 
                CL_MEM_READ_ONLY,
                imageFormat,
                size->x, size->y, size->z
        );
        data = (void *)file->data();
        ocl.queue.enqueueWriteImage(dataset, CL_FALSE, offset, region2, 0, 0, data);
        getLimits<unsigned char>(parameters, data, totalSize, &minimum, &maximum);
    } else if(typeName == "MET_FLOAT") {
        type = 3;
        file->open(rawFilename, size->x*size->y*size->z*sizeof(float));
        imageFormat = ImageFormat(CL_R, CL_FLOAT);
        dataset = Image3D(
                ocl.context, 
                CL_MEM_READ_ONLY,
                imageFormat,
                size->x, size->y, size->z
        );
        data = (void *)file->data();
        ocl.queue.enqueueWriteImage(dataset, CL_FALSE, offset, region2, 0, 0, data);
        getLimits<float>(parameters, data, totalSize, &minimum, &maximum);
    } else {
    	std::string str = "unsupported data type " + typeName;
    	throw SIPL::SIPLException(str.c_str(), __LINE__, __FILE__);
    }


    std::cout << "Dataset of size " << size->x << " " << size->y << " " << size->z << " loaded" << std::endl;
    if(getParamBool(parameters, "timing")) {
        ocl.queue.enqueueMarker(&endEvent);
        ocl.queue.finish();
        startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
        endEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &end);
        std::cout << "RUNTIME of data transfer to device: " << (end-start)*1.0e-6 << " ms" << std::endl;
        ocl.queue.enqueueMarker(&startEvent);
    }
    // Perform cropping if required
    std::string cropping = getParamStr(parameters, "cropping");
    SIPL::int3 shiftVector;
    if(cropping == "lung" || cropping == "threshold") {
        std::cout << "performing cropping" << std::endl;
        Kernel cropDatasetKernel;
        int minScanLines;
        std::string cropping_start_z;
        if(cropping == "lung") {
			cropDatasetKernel = Kernel(ocl.program, "cropDatasetLung");
			minScanLines = getParam(parameters, "min-scan-lines-lung");
			cropping_start_z = "middle";
			cropDatasetKernel.setArg(3, type);
        } else if(cropping == "threshold") {
        	cropDatasetKernel = Kernel(ocl.program, "cropDatasetThreshold");
			minScanLines = getParam(parameters, "min-scan-lines-threshold");
			cropDatasetKernel.setArg(3, getParam(parameters, "cropping-threshold"));
			cropDatasetKernel.setArg(4, type);
			cropping_start_z = getParamStr(parameters, "cropping-start-z");
        }

        Buffer scanLinesInsideX = Buffer(ocl.context, CL_MEM_WRITE_ONLY, sizeof(short)*size->x);
        Buffer scanLinesInsideY = Buffer(ocl.context, CL_MEM_WRITE_ONLY, sizeof(short)*size->y);
        Buffer scanLinesInsideZ = Buffer(ocl.context, CL_MEM_WRITE_ONLY, sizeof(short)*size->z);
        cropDatasetKernel.setArg(0, dataset);
        cropDatasetKernel.setArg(1, scanLinesInsideX);
        cropDatasetKernel.setArg(2, 0);
        ocl.queue.enqueueNDRangeKernel(
            cropDatasetKernel,
            NullRange,
            NDRange(size->x),
            NullRange
        );
        cropDatasetKernel.setArg(1, scanLinesInsideY);
        cropDatasetKernel.setArg(2, 1);
        ocl.queue.enqueueNDRangeKernel(
            cropDatasetKernel,
            NullRange,
            NDRange(size->y),
            NullRange
        );
        cropDatasetKernel.setArg(1, scanLinesInsideZ);
        cropDatasetKernel.setArg(2, 2);
        ocl.queue.enqueueNDRangeKernel(
            cropDatasetKernel,
            NullRange,
            NDRange(size->z),
            NullRange
        );
        short * scanLinesX = new short[size->x];
        short * scanLinesY = new short[size->y];
        short * scanLinesZ = new short[size->z];
        ocl.queue.enqueueReadBuffer(scanLinesInsideX, CL_FALSE, 0, sizeof(short)*size->x, scanLinesX);
        ocl.queue.enqueueReadBuffer(scanLinesInsideY, CL_FALSE, 0, sizeof(short)*size->y, scanLinesY);
        ocl.queue.enqueueReadBuffer(scanLinesInsideZ, CL_FALSE, 0, sizeof(short)*size->z, scanLinesZ);

        int x1 = 0,x2 = size->x,y1 = 0,y2 = size->y,z1 = 0,z2 = size->z;
        ocl.queue.finish();
        int startSlice, a;
		if(cropping_start_z == "middle") {
			startSlice = size->z / 2;
			a = -1;
		} else {
			startSlice = 0;
			a = 1;
		}

#pragma omp parallel sections
{
#pragma omp section
{
        for(int sliceNr = 0; sliceNr < size->x; sliceNr++) {
            if(scanLinesX[sliceNr] > minScanLines) {
                x1 = sliceNr;
                break;
            }
        }
}

#pragma omp section
{
        for(int sliceNr = size->x-1; sliceNr > 0; sliceNr--) {
            if(scanLinesX[sliceNr] > minScanLines) {
                x2 = sliceNr;
                break;
            }
        }
}
#pragma omp section
{
        for(int sliceNr = 0; sliceNr < size->y; sliceNr++) {
            if(scanLinesY[sliceNr] > minScanLines) {
                y1 = sliceNr;
                break;
            }
        }
}
#pragma omp section
{
        for(int sliceNr = size->y-1; sliceNr > 0; sliceNr--) {
            if(scanLinesY[sliceNr] > minScanLines) {
                y2 = sliceNr;
                break;
            }
        }
}

#pragma omp section
{
		for(int sliceNr = startSlice; sliceNr < size->z; sliceNr++) {
            if(a*scanLinesZ[sliceNr] > a*minScanLines) {
                z2 = sliceNr;
                break;
            }
        }
}
#pragma omp section
{
        for(int sliceNr = size->z - startSlice - 1; sliceNr > 0; sliceNr--) {
            if(a*scanLinesZ[sliceNr] > a*minScanLines) {
                z1 = sliceNr;
                break;
            }
        }
}
}
		if(cropping_start_z == "end") {
			int tmp = z1;
			z1 = z2;
			z2 = tmp;
		}

        delete[] scanLinesX;
        delete[] scanLinesY;
        delete[] scanLinesZ;

        int SIZE_X = x2-x1;
        int SIZE_Y = y2-y1;
        int SIZE_Z = z2-z1;
        if(SIZE_X == 0 || SIZE_Y == 0 || SIZE_Z == 0) {
        	char * str = new char[255];
        	sprintf(str, "Invalid cropping to new size %d, %d, %d", SIZE_X, SIZE_Y, SIZE_Z);
        	throw SIPL::SIPLException(str, __LINE__, __FILE__);
        }
	    // Make them dividable by 4
	    bool lower = false;
	    while(SIZE_X % 4 != 0 && SIZE_X < size->x) {
            if(lower && x1 > 0) {
                x1--;
            } else if(x2 < size->x) {
                x2++;
            }
            lower = !lower;
            SIZE_X = x2-x1;
	    }
	    if(SIZE_X % 4 != 0) {
			while(SIZE_X % 4 != 0)
				SIZE_X--;
	    }
	    while(SIZE_Y % 4 != 0 && SIZE_Y < size->y) {
            if(lower && y1 > 0) {
                y1--;
            } else if(y2 < size->y) {
                y2++;
            }
            lower = !lower;
            SIZE_Y = y2-y1;
	    }
	    if(SIZE_Y % 4 != 0) {
			while(SIZE_Y % 4 != 0)
				SIZE_Y--;
	    }
	    while(SIZE_Z % 4 != 0 && SIZE_Z < size->z) {
            if(lower && z1 > 0) {
                z1--;
            } else if(z2 < size->z) {
                z2++;
            }
            lower = !lower;
            SIZE_Z = z2-z1;
	    }
	    if(SIZE_Z % 4 != 0) {
			while(SIZE_Z % 4 != 0)
				SIZE_Z--;
	    }
        size->x = SIZE_X;
        size->y = SIZE_Y;
        size->z = SIZE_Z;
 

        std::cout << "Dataset cropped to " << SIZE_X << ", " << SIZE_Y << ", " << SIZE_Z << std::endl;
        Image3D imageHUvolume = Image3D(ocl.context, CL_MEM_READ_ONLY, imageFormat, SIZE_X, SIZE_Y, SIZE_Z);

        cl::size_t<3> offset;
        offset[0] = 0;
        offset[1] = 0;
        offset[2] = 0;
        cl::size_t<3> region;
        region[0] = SIZE_X;
        region[1] = SIZE_Y;
        region[2] = SIZE_Z;
        cl::size_t<3> srcOffset;
        srcOffset[0] = x1;
        srcOffset[1] = y1;
        srcOffset[2] = z1;
        shiftVector.x = x1;
        shiftVector.y = y1;
        shiftVector.z = z1;
        ocl.queue.enqueueCopyImage(dataset, imageHUvolume, srcOffset, offset, region);
        dataset = imageHUvolume;
        if(getParamBool(parameters, "timing")) {
            ocl.queue.enqueueMarker(&endEvent);
            ocl.queue.finish();
            startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
            endEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &end);
            std::cout << "Cropping time: " << (end-start)*1.0e-6 << " ms" << std::endl;
            ocl.queue.enqueueMarker(&startEvent);
        }
    } else {// End cropping
        // If cropping is not done, shrink volume so that each dimension is dividable by 4
    	bool notDividable = false;
    	if(size->x % 4 != 0 || size->y % 4 != 0 || size->z % 4 != 0)
    		notDividable = true;

    	if(notDividable) {
			while(size->x % 4 != 0)
				size->x--;
			while(size->y % 4 != 0)
				size->y--;
			while(size->z % 4 != 0)
				size->z--;

			cl::size_t<3> offset;
			offset[0] = 0;
			offset[1] = 0;
			offset[2] = 0;
			cl::size_t<3> region;
			region[0] = size->x;
			region[1] = size->y;
			region[2] = size->z;
			Image3D imageHUvolume = Image3D(ocl.context, CL_MEM_READ_ONLY, imageFormat, size->x, size->y, size->z);

			ocl.queue.enqueueCopyImage(dataset, imageHUvolume, offset, offset, region);
			dataset = imageHUvolume;

			std::cout << "NOTE: reduced size to " << size->x << ", " << size->y << ", " << size->z << std::endl;
    	}
    }
    output->setShiftVector(shiftVector);
    output->setSpacing(spacing);

    // Run toFloat kernel

    Kernel toFloatKernel = Kernel(ocl.program, "toFloat");
    Image3D convertedDataset = Image3D(
        ocl.context,
        CL_MEM_READ_ONLY,
        ImageFormat(CL_R, CL_FLOAT),
        size->x, size->y, size->z
    );

	const bool no3Dwrite = !getParamBool(parameters, "3d_write");
    if(no3Dwrite) {
        Buffer convertedDatasetBuffer = Buffer(
                ocl.context, 
                CL_MEM_WRITE_ONLY,
                sizeof(float)*size->x*size->y*size->z
        );
        toFloatKernel.setArg(0, dataset);
        toFloatKernel.setArg(1, convertedDatasetBuffer);
        toFloatKernel.setArg(2, minimum);
        toFloatKernel.setArg(3, maximum);
        toFloatKernel.setArg(4, type);

        ocl.queue.enqueueNDRangeKernel(
            toFloatKernel,
            NullRange,
            NDRange(size->x, size->y, size->z),
            NullRange
        );

        cl::size_t<3> offset;
        offset[0] = 0;
        offset[1] = 0;
        offset[2] = 0;
        cl::size_t<3> region;
        region[0] = size->x;
        region[1] = size->y;
        region[2] = size->z;

        ocl.queue.enqueueCopyBufferToImage(
                convertedDatasetBuffer, 
                convertedDataset, 
                0,
                offset,
                region
        );
    } else {
        toFloatKernel.setArg(0, dataset);
        toFloatKernel.setArg(1, convertedDataset);
        toFloatKernel.setArg(2, minimum);
        toFloatKernel.setArg(3, maximum);
        toFloatKernel.setArg(4, type);

        ocl.queue.enqueueNDRangeKernel(
            toFloatKernel,
            NullRange,
            NDRange(size->x, size->y, size->z),
            NullRange
        );
    }
    if(getParamBool(parameters, "timing")) {
        ocl.queue.enqueueMarker(&endEvent);
        ocl.queue.finish();
        startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
        endEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &end);
        std::cout << "RUNTIME of to float conversion: " << (end-start)*1.0e-6 << " ms" << std::endl;
        ocl.queue.enqueueMarker(&startEvent);
    }

    dataset.setDestructorCallback((void (__stdcall *)(cl_mem,void *))unmapRawfile, (void *)(file));
    // Return dataset
    return convertedDataset;
}

TSFOutput::TSFOutput(OpenCL * ocl, SIPL::int3 * size, bool TDFis16bit) {
	this->TDFis16bit = TDFis16bit;
	this->ocl = ocl;
	this->size = size;
	hostHasCenterlineVoxels = false;
	hostHasSegmentation = false;
	hostHasTDF = false;
	deviceHasCenterlineVoxels = false;
	deviceHasSegmentation = false;
	deviceHasTDF = false;
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
