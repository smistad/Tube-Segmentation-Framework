#include "tube-segmentation.hpp"
#include "SIPL/Exceptions.hpp"

using namespace cl;

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
