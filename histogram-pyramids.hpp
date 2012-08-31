#ifndef HP_H
#define HP_H

#include "commons.hpp"
#include <vector>

class HistogramPyramid {
    public:
        int getSum();
        virtual cl::Buffer createPositionBuffer() = 0;
    protected:
        OpenCL ocl;
        int size;
        int sum;
};

class HistogramPyramid2D : public HistogramPyramid {
    public:
        HistogramPyramid2D(OpenCL);
        void create(cl::Image2D, int, int);
        cl::Buffer createPositionBuffer();
        void traverse(cl::Kernel, int);
    private:
        std::vector<cl::Image2D> HPlevels;
};

class HistogramPyramid3D : public HistogramPyramid {
    public:
        HistogramPyramid3D(OpenCL);
        void create(cl::Image3D, int, int, int);
        cl::Buffer createPositionBuffer();
        void traverse(cl::Kernel, int);
    private:
        std::vector<cl::Image3D> HPlevels;
};

class HistogramPyramid3DBuffer : public HistogramPyramid {
    public:
        HistogramPyramid3DBuffer(OpenCL);
        void create(cl::Buffer, int, int, int);
        cl::Buffer createPositionBuffer();
        void traverse(cl::Kernel, int);
    private:
        std::vector<cl::Buffer> HPlevels;
};

std::string insertHPOpenCLCode(std::string, int);

#endif
