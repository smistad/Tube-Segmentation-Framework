#include "segmentation.hpp"
using namespace cl;

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

