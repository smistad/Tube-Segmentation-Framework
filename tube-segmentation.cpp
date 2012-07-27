#include "tube-segmentation.hpp"
#include "SIPL/Exceptions.hpp"

using namespace cl;

template <typename T>
T getParamf(std::map<std::string, std::string> parameters, std::string parameterName, T defaultValue) {
    if(parameters.count(parameterName) == 1) {
        return atof(parameters[parameterName].c_str());
    } else {
        return defaultValue;
    }
}

template <typename T>
T getParami(std::map<std::string, std::string> parameters, std::string parameterName, T defaultValue) {
    if(parameters.count(parameterName) == 1) {
        return atoi(parameters[parameterName].c_str());
    } else {
        return defaultValue;
    }
}


TubeSegmentation runCircleFittingMethod(OpenCL, Image3D dataset, SIPL::int3 size, std::map<std::string, std::string> parameters) {
    // Set up parameters
    int GVFIterations = getParami(parameters, "gvf-iterations", 250);
    float radiusMin = getParamf(parameters, "radius-min", 0.5);
    float radiusMax = getParamf(parameters, "radius-min", 15.0);
    float radiusStep = getParamf(parameters, "radius-step", 0.5);

    // Create kernels
    Kernel blurVolumeWithGaussianKernel(program, "blurVolumeWithGaussian");
    Kernel createVectorFieldKernel(program, "createVectorField");
    Kernel circleFittingTDFKernel(program, "circleFittingTDF");
    Kernel dilateKernel = Kernel(program, "dilate");
    Kernel erodeKernel = Kernel(program, "erode");
    Kernel initGrowKernel = Kernel(program, "initGrowing");
    Kernel growKernel = Kernel(program, "grow");

    int maskSize = 0;
    float * mask;// = createBlurMask(0.5, &maskSize);
    Buffer blurMask;

    /*
    // Run blurVolumeWithGaussian on processedVolume
    blurVolumeWithGaussianKernel.setArg(0, imageVolume);
    blurVolumeWithGaussianKernel.setArg(1, blurredVolume);
    blurVolumeWithGaussianKernel.setArg(2, maskSize);
    blurVolumeWithGaussianKernel.setArg(3, blurMask);
    queue.enqueueNDRangeKernel(
            blurVolumeWithGaussianKernel,
            NullRange,
            NDRange(SIZE_X,SIZE_Y,SIZE_Z),
            NDRange(4,4,4)
    );

    // Copy buffer to image
    queue.enqueueCopyBufferToImage(blurredVolume, imageVolume, 0, offset, region);
    */

#ifdef TIMING
    queue.enqueueMarker(&startEvent);
#endif
    Image3D vectorField = Image3D(context, CL_MEM_READ_WRITE, ImageFormat(CL_RGBA, CL_SNORM_INT16), SIZE_X, SIZE_Y, SIZE_Z);
    
    // Run create vector field
    createVectorFieldKernel.setArg(0, imageVolume);
    createVectorFieldKernel.setArg(1, vectorField);
    createVectorFieldKernel.setArg(2, Fmax);

    queue.enqueueNDRangeKernel(
            createVectorFieldKernel,
            NullRange,
            NDRange(SIZE_X,SIZE_Y,SIZE_Z),
            NDRange(4,4,4)
    );
    
#ifdef TIMING
    queue.enqueueMarker(&endEvent);
    queue.finish();
    startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    endEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &end);
    std::cout << "RUNTIME of Create vector field: " << (end-start)*1.0e-6 << " ms" << std::endl;
#endif
#ifdef TIMING
    queue.enqueueMarker(&startEvent);
#endif
    // Run circle fitting TDF kernel
    Buffer bufferT = Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*SIZE_X*SIZE_Y*SIZE_Z);
    Buffer bufferRadius = Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*SIZE_X*SIZE_Y*SIZE_Z);
    circleFittingTDFKernel.setArg(0, vectorField);
    circleFittingTDFKernel.setArg(1, bufferT);
    circleFittingTDFKernel.setArg(2, bufferRadius);
    circleFittingTDFKernel.setArg(3, rMin);
    circleFittingTDFKernel.setArg(4, 3.0f);
    circleFittingTDFKernel.setArg(5, 0.5f);

    queue.enqueueNDRangeKernel(
            circleFittingTDFKernel,
            NullRange,
            NDRange(SIZE_X,SIZE_Y,SIZE_Z),
            NDRange(4,4,4)
    );
    // Transfer buffer back to host
    float * Tsmall = new float[SIZE_X*SIZE_Y*SIZE_Z];
    float * Radiussmall = new float[SIZE_X*SIZE_Y*SIZE_Z];
    queue.enqueueReadBuffer(bufferT, CL_FALSE, 0, sizeof(float)*SIZE_X*SIZE_Y*SIZE_Z, Tsmall);
    queue.enqueueReadBuffer(bufferRadius, CL_FALSE, 0, sizeof(float)*SIZE_X*SIZE_Y*SIZE_Z, Radiussmall);
#ifdef TIMING
    queue.enqueueMarker(&endEvent);
    queue.finish();
    startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    endEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &end);
    std::cout << "RUNTIME of TDF small: " << (end-start)*1.0e-6 << " ms" << std::endl;
#endif
    /* Large Airways */
    
#ifdef TIMING
    queue.enqueueMarker(&startEvent);
#endif
    Image3D blurredVolume = Image3D(context, CL_MEM_READ_WRITE, ImageFormat(CL_R, CL_FLOAT), SIZE_X, SIZE_Y, SIZE_Z);

    // Put processedVolume buffer into image
    mask = createBlurMask(1.0, &maskSize);
    blurMask = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*(maskSize*2+1)*(maskSize*2+1)*(maskSize*2+1), mask);
    // Run blurVolumeWithGaussian on processedVolume
    blurVolumeWithGaussianKernel.setArg(0, imageVolume);
    blurVolumeWithGaussianKernel.setArg(1, blurredVolume);
    blurVolumeWithGaussianKernel.setArg(2, maskSize);
    blurVolumeWithGaussianKernel.setArg(3, blurMask);

    queue.enqueueNDRangeKernel(
            blurVolumeWithGaussianKernel,
            NullRange,
            NDRange(SIZE_X,SIZE_Y,SIZE_Z),
            NDRange(4,4,4)
    );

#ifdef TIMING
    queue.enqueueMarker(&endEvent);
    queue.finish();
    startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    endEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &end);
    std::cout << "RUNTIME blurring: " << (end-start)*1.0e-6 << " ms" << std::endl;
#endif
#ifdef TIMING
    queue.enqueueMarker(&startEvent);
#endif
    
    // Run create vector field
    createVectorFieldKernel.setArg(0, blurredVolume);
    createVectorFieldKernel.setArg(1, vectorField);
    createVectorFieldKernel.setArg(2, Fmax);

    queue.enqueueNDRangeKernel(
            createVectorFieldKernel,
            NullRange,
            NDRange(SIZE_X,SIZE_Y,SIZE_Z),
            NDRange(4,4,4)
    );
    
#ifdef TIMING
    queue.enqueueMarker(&endEvent);
    queue.finish();
    startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    endEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &end);
    std::cout << "RUNTIME Create vector field: " << (end-start)*1.0e-6 << " ms" << std::endl;
#endif
#ifdef TIMING
    queue.enqueueMarker(&startEvent);
#endif

    // Run GVF on iVectorField as initial vector field
    Kernel GVFInitKernel = Kernel(program, "GVF3DInit");
    Kernel GVFIterationKernel = Kernel(program, "GVF3DIteration");
    Kernel GVFFinishKernel = Kernel(program, "GVF3DFinish");

    Image3D vectorField1 = Image3D(context, CL_MEM_READ_WRITE, ImageFormat(CL_RGBA, CL_SNORM_INT16), SIZE_X, SIZE_Y, SIZE_Z);
    Image3D initVectorField = Image3D(context, CL_MEM_READ_WRITE, ImageFormat(CL_RG, CL_SNORM_INT16), SIZE_X, SIZE_Y, SIZE_Z);
    std::cout << "Running GVF... ( " << ITERATIONS << " )" << std::endl; 

    // init vectorField from image
    GVFInitKernel.setArg(0, vectorField);
    GVFInitKernel.setArg(1, vectorField1);
    GVFInitKernel.setArg(2, initVectorField);
    queue.enqueueNDRangeKernel(
            GVFInitKernel,
            NullRange,
            NDRange(SIZE_X,SIZE_Y,SIZE_Z),
            NDRange(4,4,4)
    );
    // Run iterations
    GVFIterationKernel.setArg(0, initVectorField);
    GVFIterationKernel.setArg(3, MU);

    for(int i = 0; i < ITERATIONS; i++) {
        if(i % 2 == 0) {
            GVFIterationKernel.setArg(1, vectorField1);
            GVFIterationKernel.setArg(2, vectorField);
        } else {
            GVFIterationKernel.setArg(1, vectorField);
            GVFIterationKernel.setArg(2, vectorField1);
        }
            queue.enqueueNDRangeKernel(
                    GVFIterationKernel,
                    NullRange,
                    NDRange(SIZE_X,SIZE_Y,SIZE_Z),
                    NDRange(4,4,4)
            );
    }

    // Copy vector field to image
    GVFFinishKernel.setArg(0, vectorField1);
    GVFFinishKernel.setArg(1, vectorField);

    queue.enqueueNDRangeKernel(
            GVFFinishKernel,
            NullRange,
            NDRange(SIZE_X,SIZE_Y,SIZE_Z),
            NDRange(4,4,4)
    );
#ifdef TIMING
    queue.enqueueMarker(&endEvent);
    queue.finish();
    startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    endEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &end);
    std::cout << "RUNTIME of GVF: " << (end-start)*1.0e-6 << " ms" << std::endl;
#endif

#ifdef TIMING
    queue.enqueueMarker(&startEvent);
#endif
    // Run circle fitting TDF kernel on GVF result
    circleFittingTDFKernel.setArg(0, vectorField);
    circleFittingTDFKernel.setArg(1, bufferT);
    circleFittingTDFKernel.setArg(2, bufferRadius);
    circleFittingTDFKernel.setArg(3, 1.0f);
    circleFittingTDFKernel.setArg(4, rMax);
    circleFittingTDFKernel.setArg(5, 1.0f);

    queue.enqueueNDRangeKernel(
            circleFittingTDFKernel,
            NullRange,
            NDRange(SIZE_X,SIZE_Y,SIZE_Z),
            NDRange(4,4,4)
    );

#ifdef TIMING
    queue.enqueueMarker(&endEvent);
    queue.finish();
    startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    endEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &end);
    std::cout << "RUNTIME of TDF large: " << (end-start)*1.0e-6 << " ms" << std::endl;
#endif
    START_TIMER
    // Transfer buffer back to host
    short * Fs = new short[SIZE_X*SIZE_Y*SIZE_Z*4];
    queue.enqueueReadImage(vectorField, CL_TRUE, offset, region, 0, 0, Fs);
    TubeSegmentation TS;
    TS.Fx = new float[SIZE_X*SIZE_Y*SIZE_Z];
    TS.Fy = new float[SIZE_X*SIZE_Y*SIZE_Z];
    TS.Fz = new float[SIZE_X*SIZE_Y*SIZE_Z];
#pragma omp parallel for
    for(int i = 0; i < SIZE_X*SIZE_Y*SIZE_Z; i++) {
        TS.Fx[i] = MAX(-1.0f, Fs[i*4] / 32767.0f);
        TS.Fy[i] = MAX(-1.0f, Fs[i*4+1] / 32767.0f);;
        TS.Fz[i] = MAX(-1.0f, Fs[i*4+2] / 32767.0f);
    }
    delete[] Fs;
    float * Tlarge = new float[SIZE_X*SIZE_Y*SIZE_Z];
    float * Radiuslarge = new float[SIZE_X*SIZE_Y*SIZE_Z];
    queue.enqueueReadBuffer(bufferT, CL_TRUE, 0, sizeof(float)*SIZE_X*SIZE_Y*SIZE_Z, Tlarge);
    queue.enqueueReadBuffer(bufferRadius, CL_TRUE, 0, sizeof(float)*SIZE_X*SIZE_Y*SIZE_Z, Radiuslarge);

    float * Tmerged = new float[SIZE_X*SIZE_Y*SIZE_Z];
    float * Radiusmerged = new float[SIZE_X*SIZE_Y*SIZE_Z];
#pragma omp parallel for
    for(int i = 0; i < SIZE_X*SIZE_Y*SIZE_Z; i++) {
        if(Tsmall[i] < Tlarge[i]) {
            Tmerged[i] = Tlarge[i];
            Radiusmerged[i] = Radiuslarge[i];
        } else {
            Tmerged[i] = Tsmall[i];
            Radiusmerged[i] = Radiussmall[i];
        }
    }
    TS.TDF = Tmerged;
    TS.radius = Radiusmerged;
    delete[] Tlarge;
    delete[] Tsmall;
    delete[] Radiussmall;
    delete[] Radiuslarge;
    
    std::stack<CenterlinePoint> centerlineStack;
    TS.centerline = runCenterlineExtraction(T, 0.6, SIZE_X, SIZE_Y, SIZE_Z, &centerlineStack);

    // Dilate the centerline
    volume = Image3D(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, ImageFormat(CL_R, CL_SIGNED_INT8), SIZE_X, SIZE_Y, SIZE_Z, 0, 0, TS.centerline);
    Image3D volume2 = Image3D(context, CL_MEM_READ_WRITE, ImageFormat(CL_R, CL_SIGNED_INT8), SIZE_X, SIZE_Y, SIZE_Z);
    queue.enqueueCopyImage(volume, volume2, offset, offset, region);
#ifdef TIMING
    queue.finish();
    STOP_TIMER("Centerline extraction + transfer of data back and forth")
    queue.enqueueMarker(&startEvent);
#endif

    initGrowKernel.setArg(0, volume);
    initGrowKernel.setArg(1, volume2);
    queue.enqueueNDRangeKernel(
        initGrowKernel,
        NullRange,
        NDRange(SIZE_X, SIZE_Y, SIZE_Z),
        NDRange(4,4,4)
    );
        
    
    // Do the segmentation here on the segmentation data (volume)
    int stopGrowing = 0;
    Buffer stop = Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), &stopGrowing);
    
    growKernel.setArg(1, vectorField);	
    growKernel.setArg(3, stop);

    int i = 0;
    while(stopGrowing == 0) {
        // run for at least 10 iterations
        if(i > 10) {
            stopGrowing = 1;
            queue.enqueueWriteBuffer(stop, CL_TRUE, 0, sizeof(int), &stopGrowing);
        }
        if(i % 2 == 0) {
            growKernel.setArg(0, volume);
            growKernel.setArg(2, volume2);
        } else {
            growKernel.setArg(0, volume2);
            growKernel.setArg(2, volume);
        }

        queue.enqueueNDRangeKernel(
                growKernel,
                NullRange,
                    NDRange(SIZE_X, SIZE_Y, SIZE_Z),
                    NDRange(4,4,4)
                );
        if(i > 10)
            queue.enqueueReadBuffer(stop, CL_TRUE, 0, sizeof(int), &stopGrowing);
        i++;
    }
    std::cout << "segmentation result grown in " << i << " iterations" << std::endl;

    dilateKernel.setArg(0, volume);
    dilateKernel.setArg(1, volume2);
   
    queue.enqueueNDRangeKernel(
        dilateKernel,
        NullRange,
        NDRange(SIZE_X, SIZE_Y, SIZE_Z),
        NDRange(4,4,4)
    );

    erodeKernel.setArg(0, volume2);
    erodeKernel.setArg(1, volume);
   
    queue.enqueueNDRangeKernel(
        erodeKernel,
        NullRange,
        NDRange(SIZE_X, SIZE_Y, SIZE_Z),
        NDRange(4,4,4)
    );

    TS.segmentation= new char[SIZE_X*SIZE_Y*SIZE_Z];
    queue.enqueueReadImage(volume, CL_TRUE, offset, region, 0, 0, TS.segmentation);
#ifdef TIMING
    queue.enqueueMarker(&endEvent);
    queue.finish();
    startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    endEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &end);
    std::cout << "RUNTIME of segmentation: " << (end-start)*1.0e-6 << " ms" << std::endl;
#endif
    START_TIMER
    writeToRaw<char>(TS.centerline, storageDirectory + "centerline.raw", SIZE_X, SIZE_Y, SIZE_Z);
    writeToRaw<char>(TS.segmentation, storageDirectory + "segmentation.raw", SIZE_X, SIZE_Y, SIZE_Z);
    STOP_TIMER("writing segmentation and centerline to disk")
}

std::map<std::string, std::string> getParameters(int argc, char ** argv) {
    std::map<std::string, std::string> parameters;
    // Go through each parameter, first parameter is filename
    for(int i = 2; i < argc; i++) {
        std::string token = argv[i];
        if(token.substr(0,2) == "--") {
            // Check to see if the parameter has a value
            std::string nextToken;
            if(i+1 < argc) {
                nextToken = argv[i+1];
            } else {
                nextToken = "--";
            }
            if(nextToken.substr(0,2) == "--") {
                // next token is not a value
                parameters[token.substr(2)] = "dummy-value";
            } else {
                // next token is a value, store the value
                parameters[token.substr(2)] = nextToken;
                i++;
            }
        }
    }

    return parameters;
}

Image3D readDatasetAndTransfer(OpenCL ocl, std::string filename, std::map<std::string, std::string> parameters, SIPL::int3 * size) {
    // Read mhd file, determine file type
    std::fstream mhdFile;
    mhdFile.open(filename.c_str(), std::fstream::in);
    std::string typeName = "";
    do {
        std::string line;
        std::getline(mhdFile, line);
        if(line.substr(0, 11) == "ElementType") 
            typeName = line.substr(11+3);
    } while(!mhdFile.eof());

    if(typeName == "") 
        throw SIPL::SIPLException("no data type defined in MHD file");

    // Read dataset using SIPL and transfer to device
    Image3D dataset;
    if(typeName == "MET_SHORT") {
        SIPL::Volume<short> * v = new SIPL::Volume<short>(filename);
        dataset = Image3D(
                ocl.context, 
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                ImageFormat(CL_R, CL_SIGNED_INT16),
                v->getWidth(), v->getHeight(), v->getDepth(),
                0, 0, v->getData()
        );
        size->x = v->getWidth();
        size->y = v->getHeight();
        size->z = v->getDepth();
        delete v;
    } else if(typeName == "MET_USHORT") {
        SIPL::Volume<SIPL::ushort> * v = new SIPL::Volume<SIPL::ushort>(filename);
        dataset = Image3D(
                ocl.context, 
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                ImageFormat(CL_R, CL_UNSIGNED_INT16),
                v->getWidth(), v->getHeight(), v->getDepth(),
                0, 0, v->getData()
        );
        size->x = v->getWidth();
        size->y = v->getHeight();
        size->z = v->getDepth();
        delete v;
    } else if(typeName == "MET_CHAR") {
        SIPL::Volume<char> * v = new SIPL::Volume<char>(filename);
        dataset = Image3D(
                ocl.context, 
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                ImageFormat(CL_R, CL_SIGNED_INT8),
                v->getWidth(), v->getHeight(), v->getDepth(),
                0, 0, v->getData()
        );
        size->x = v->getWidth();
        size->y = v->getHeight();
        size->z = v->getDepth();
        delete v;
    } else if(typeName == "MET_UCHAR") {
        SIPL::Volume<SIPL::uchar> * v = new SIPL::Volume<SIPL::uchar>(filename);
        dataset = Image3D(
                ocl.context, 
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                ImageFormat(CL_R, CL_UNSIGNED_INT8),
                v->getWidth(), v->getHeight(), v->getDepth(),
                0, 0, v->getData()
        );
        size->x = v->getWidth();
        size->y = v->getHeight();
        size->z = v->getDepth();
        delete v;
    } else if(typeName == "MET_FLOAT") {
        SIPL::Volume<float> * v = new SIPL::Volume<float>(filename);
        dataset = Image3D(
                ocl.context, 
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                ImageFormat(CL_R, CL_FLOAT),
                v->getWidth(), v->getHeight(), v->getDepth(),
                0, 0, v->getData()
        );
        size->x = v->getWidth();
        size->y = v->getHeight();
        size->z = v->getDepth();
        delete v;
    } else {
        std::string msg = "unsupported filetype " + typeName;
        throw SIPL::SIPLException(msg.c_str());
    }

    // Perform cropping if required
    if(parameters.count("cropping") == 1) {
        // TODO: perform cropping
    }

    // Run toFloat kernel
    float minimum = 0.0f, maximum = 1.0f;
    if(parameters.count("minimum") == 1)
        minimum = atof(parameters["minimum"].c_str());
    
    if(parameters.count("maximum") == 1)
        maximum = atof(parameters["maximum"].c_str());

    Kernel toFloatKernel = Kernel(ocl.program, "toFloat");
    Image3D * convertedDataset = new Image3D(
        ocl.context,
        CL_MEM_READ_ONLY,
        ImageFormat(CL_R, CL_FLOAT),
        size->x, size->y, size->z
    );

    toFloatKernel.setArg(0, dataset);
    toFloatKernel.setArg(1, *convertedDataset);
    toFloatKernel.setArg(2, minimum);
    toFloatKernel.setArg(3, maximum);

    ocl.queue.enqueueNDRangeKernel(
        toFloatKernel,
        NullRange,
        NDRange(size->x, size->y, size->z),
        NullRange
    );

    // Return dataset
    return *convertedDataset;
}
