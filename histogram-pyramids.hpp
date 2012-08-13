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
        void update(cl::Image2D);
        void update();
    private:
        std::vector<cl::Image2D> HPlevels;
};

class HistogramPyramid3D : public HistogramPyramid {
    public:
        HistogramPyramid3D(OpenCL);
        void create(cl::Image3D, int, int, int);
        cl::Buffer createPositionBuffer();
        void traverse(cl::Kernel, int);
        void update(cl::Image3D);
        void update();
    private:
        std::vector<cl::Image3D> HPlevels;
};

std::string insertHPOpenCLCode(std::string, int);

#endif
