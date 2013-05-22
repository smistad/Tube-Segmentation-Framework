#include "tube-segmentation.hpp"
#include "SIPL/Types.hpp"
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
#ifdef CPP11
#include <unordered_set>
using std::unordered_set;
#else
#include <boost/unordered_set.hpp>
using boost::unordered_set;
#endif
#include "histogram-pyramids.hpp"
//#include "tsf-config.h"

//#define TIMING
#ifdef CPP11
#include <chrono>
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
#define INIT_TIMER high_resolution_clock::time_point timerStart = high_resolution_clock::now();
#define START_TIMER  timerStart = high_resolution_clock::now();
#define STOP_TIMER(name)  std::cout << "RUNTIME of " << name << ": " << \
        duration_cast<milliseconds>( \
                            high_resolution_clock::now()-timerStart \
                    ).count() << " ms " << std::endl;
#else
#define INIT_TIMER
#define START_TIMER
#define STOP_TIMER(name)
#endif


// Undefine windows crap
#ifdef WIN32
#undef min
#undef max
#else
#define __stdcall
#endif

#define MAX(a,b) a > b ? a : b

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

TSFGarbageCollector * GC;
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
    GC = new TSFGarbageCollector;
    TSFOutput * output = new TSFOutput(ocl, size, getParamBool(parameters, "16bit-vectors"));
    try {
        // Read dataset and transfer to device
        cl::Image3D * dataset = new cl::Image3D;
        GC->addMemObject(dataset);
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
        GC->deleteAllMemObjects();
        delete GC;
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
    GC->deleteAllMemObjects();
    delete GC;
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
    std::cout << "object was deleted" << std::endl;
}



typedef struct CenterlinePoint {
    int3 pos;
    bool large;
    CenterlinePoint * next;
} CenterlinePoint;

typedef struct point {
    float value;
    int x,y,z;
} point;

class PointComparison {
    public:
    bool operator() (const point &lhs, const point &rhs) const {
        return (lhs.value < rhs.value);
    }
};

bool inBounds(SIPL::int3 pos, SIPL::int3 size) {
    return pos.x > 0 && pos.y > 0 && pos.z > 0 && pos.x < size.x && pos.y < size.y && pos.z < size.z;
}

#define LPOS(a,b,c) (a)+(b)*(size.x)+(c)*(size.x*size.y)
#define M(a,b,c) 1-sqrt(pow(T.Fx[a+b*size.x+c*size.x*size.y],2.0f) + pow(T.Fy[a+b*size.x+c*size.x*size.y],2.0f) + pow(T.Fz[a+b*size.x+c*size.x*size.y],2.0f))
#define SQR_MAG(pos) sqrt(pow(T.Fx[pos.x+pos.y*size.x+pos.z*size.x*size.y],2.0f) + pow(T.Fy[pos.x+pos.y*size.x+pos.z*size.x*size.y],2.0f) + pow(T.Fz[pos.x+pos.y*size.x+pos.z*size.x*size.y],2.0f))
#define SQR_MAG_SMALL(pos) sqrt(pow(T.FxSmall[pos.x+pos.y*size.x+pos.z*size.x*size.y],2.0f) + pow(T.FySmall[pos.x+pos.y*size.x+pos.z*size.x*size.y],2.0f) + pow(T.FzSmall[pos.x+pos.y*size.x+pos.z*size.x*size.y],2.0f))

#define SIZE 3

float hypot2(float x, float y) {
  return sqrt(x*x+y*y);
}

// Symmetric Householder reduction to tridiagonal form.

void tred2(float V[SIZE][SIZE], float d[SIZE], float e[SIZE]) {

//  This is derived from the Algol procedures tred2 by
//  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
//  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
//  Fortran subroutine in EISPACK.

  for (int j = 0; j < SIZE; j++) {
    d[j] = V[SIZE-1][j];
  }

  // Householder reduction to tridiagonal form.

  for (int i = SIZE-1; i > 0; i--) {

    // Scale to avoid under/overflow.

    float scale = 0.0f;
    float h = 0.0f;
    for (int k = 0; k < i; k++) {
      scale = scale + fabs(d[k]);
    }
    if (scale == 0.0f) {
      e[i] = d[i-1];
      for (int j = 0; j < i; j++) {
        d[j] = V[i-1][j];
        V[i][j] = 0.0f;
        V[j][i] = 0.0f;
      }
    } else {

      // Generate Householder vector.

      for (int k = 0; k < i; k++) {
        d[k] /= scale;
        h += d[k] * d[k];
      }
      float f = d[i-1];
      float g = sqrt(h);
      if (f > 0) {
        g = -g;
      }
      e[i] = scale * g;
      h = h - f * g;
      d[i-1] = f - g;
      for (int j = 0; j < i; j++) {
        e[j] = 0.0f;
      }

      // Apply similarity transformation to remaining columns.

      for (int j = 0; j < i; j++) {
        f = d[j];
        V[j][i] = f;
        g = e[j] + V[j][j] * f;
        for (int k = j+1; k <= i-1; k++) {
          g += V[k][j] * d[k];
          e[k] += V[k][j] * f;
        }
        e[j] = g;
      }
      f = 0.0f;
      for (int j = 0; j < i; j++) {
        e[j] /= h;
        f += e[j] * d[j];
      }
      float hh = f / (h + h);
      for (int j = 0; j < i; j++) {
        e[j] -= hh * d[j];
      }
      for (int j = 0; j < i; j++) {
        f = d[j];
        g = e[j];
        for (int k = j; k <= i-1; k++) {
          V[k][j] -= (f * e[k] + g * d[k]);
        }
        d[j] = V[i-1][j];
        V[i][j] = 0.0f;
      }
    }
    d[i] = h;
  }

  // Accumulate transformations.

  for (int i = 0; i < SIZE-1; i++) {
    V[SIZE-1][i] = V[i][i];
    V[i][i] = 1.0f;
    float h = d[i+1];
    if (h != 0.0f) {
      for (int k = 0; k <= i; k++) {
        d[k] = V[k][i+1] / h;
      }
      for (int j = 0; j <= i; j++) {
        float g = 0.0f;
        for (int k = 0; k <= i; k++) {
          g += V[k][i+1] * V[k][j];
        }
        for (int k = 0; k <= i; k++) {
          V[k][j] -= g * d[k];
        }
      }
    }
    for (int k = 0; k <= i; k++) {
      V[k][i+1] = 0.0f;
    }
  }
  for (int j = 0; j < SIZE; j++) {
    d[j] = V[SIZE-1][j];
    V[SIZE-1][j] = 0.0f;
  }
  V[SIZE-1][SIZE-1] = 1.0f;
  e[0] = 0.0f;
}

// Symmetric tridiagonal QL algorithm.

void tql2(float V[SIZE][SIZE], float d[SIZE], float e[SIZE]) {

//  This is derived from the Algol procedures tql2, by
//  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
//  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
//  Fortran subroutine in EISPACK.

  for (int i = 1; i < SIZE; i++) {
    e[i-1] = e[i];
  }
  e[SIZE-1] = 0.0f;

  float f = 0.0f;
  float tst1 = 0.0f;
  float eps = pow(2.0f,-52.0f);
  for (int l = 0; l < SIZE; l++) {

    // Find small subdiagonal element

    tst1 = MAX(tst1,fabs(d[l]) + fabs(e[l]));
    int m = l;
    while (m < SIZE) {
      if (fabs(e[m]) <= eps*tst1) {
        break;
      }
      m++;
    }

    // If m == l, d[l] is an eigenvalue,
    // otherwise, iterate.

    if (m > l) {
      int iter = 0;
      do {
        iter = iter + 1;  // (Could check iteration count here.)

        // Compute implicit shift

        float g = d[l];
        float p = (d[l+1] - g) / (2.0f * e[l]);
        float r = hypot2(p,1.0f);
        if (p < 0) {
          r = -r;
        }
        d[l] = e[l] / (p + r);
        d[l+1] = e[l] * (p + r);
        float dl1 = d[l+1];
        float h = g - d[l];
        for (int i = l+2; i < SIZE; i++) {
          d[i] -= h;
        }
        f = f + h;

        // Implicit QL transformation.

        p = d[m];
        float c = 1.0f;
        float c2 = c;
        float c3 = c;
        float el1 = e[l+1];
        float s = 0.0f;
        float s2 = 0.0f;
        for (int i = m-1; i >= l; i--) {
          c3 = c2;
          c2 = c;
          s2 = s;
          g = c * e[i];
          h = c * p;
          r = hypot2(p,e[i]);
          e[i+1] = s * r;
          s = e[i] / r;
          c = p / r;
          p = c * d[i] - s * g;
          d[i+1] = h + s * (c * g + s * d[i]);

          // Accumulate transformation.

          for (int k = 0; k < SIZE; k++) {
            h = V[k][i+1];
            V[k][i+1] = s * V[k][i] + c * h;
            V[k][i] = c * V[k][i] - s * h;
          }
        }
        p = -s * s2 * c3 * el1 * e[l] / dl1;
        e[l] = s * p;
        d[l] = c * p;

        // Check for convergence.

      } while (fabs(e[l]) > eps*tst1);
    }
    d[l] = d[l] + f;
    e[l] = 0.0f;
  }

  // Sort eigenvalues and corresponding vectors.

  for (int i = 0; i < SIZE-1; i++) {
    int k = i;
    float p = d[i];
    for (int j = i+1; j < SIZE; j++) {
      if (fabs(d[j]) < fabs(p)) {
        k = j;
        p = d[j];
      }
    }
    if (k != i) {
      d[k] = d[i];
      d[i] = p;
      for (int j = 0; j < SIZE; j++) {
        p = V[j][i];
        V[j][i] = V[j][k];
        V[j][k] = p;
      }
    }
  }
}

void eigen_decomposition(float A[SIZE][SIZE], float V[SIZE][SIZE], float d[SIZE]) {
  float e[SIZE];
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      V[i][j] = A[i][j];
    }
  }
  tred2(V, d, e);
  tql2(V, d, e);
}

float dot(float3 a, float3 b) {
    return a.x*b.x+a.y*b.y+a.z*b.z;
}
int3 operator-(int3 a, int3 b) {
    int3 c;
    c.x = a.x-b.x;
    c.y = a.y-b.y;
    c.z = a.z-b.z;
    return c;
}

float3 normalize(float3 a) {
    float3 b;
    float sqrMag = sqrt((float)(a.x*a.x+a.y*a.y+a.z*a.z));
    b.x = a.x / sqrMag;
    b.y = a.y / sqrMag;
    b.z = a.z / sqrMag;
    return b;
}
float3 normalize(int3 a) {
    float3 b;
    float sqrMag = sqrt((float)(a.x*a.x+a.y*a.y+a.z*a.z));
    b.x = a.x / sqrMag;
    b.y = a.y / sqrMag;
    b.z = a.z / sqrMag;
    return b;
}

#define POS(pos) pos.x+pos.y*size.x+pos.z*size.x*size.y
float3 gradient(TubeSegmentation &TS, int3 pos, int volumeComponent, int dimensions, int3 size) {
    float * Fx = TS.Fx;
    float * Fy = TS.Fy;
    float * Fz = TS.Fz;
    float f100, f_100, f010, f0_10, f001, f00_1;
    int3 npos = pos;
    switch(volumeComponent) {
        case 0:

        npos.x +=1;
        f100 = Fx[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
        npos.x -=2;
        f_100 = Fx[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
        if(dimensions > 1) {
            npos = pos;
            npos.y += 1;
            f010 = Fx[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
            npos.y -= 2;
            f0_10 = Fx[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
        }
        if(dimensions > 2) {
            npos = pos;
            npos.z += 1;
            f001 = Fx[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
            npos.z -= 2;
            f00_1 =Fx[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
        }
    break;
        case 1:

        npos.x +=1;
        f100 = Fy[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
        npos.x -=2;
        f_100 = Fy[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
        if(dimensions > 1) {
            npos = pos;
            npos.y += 1;
            f010 = Fy[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
            npos.y -= 2;
            f0_10 = Fy[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
        }
        if(dimensions > 2) {
            npos = pos;
            npos.z += 1;
            f001 = Fy[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
            npos.z -= 2;
            f00_1 =Fy[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
        }
    break;
        case 2:

        npos.x +=1;
        f100 = Fz[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
        npos.x -=2;
        f_100 = Fz[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
        if(dimensions > 1) {
            npos = pos;
            npos.y += 1;
            f010 = Fz[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
            npos.y -= 2;
            f0_10 = Fz[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
        }
        if(dimensions > 2) {
            npos = pos;
            npos.z += 1;
            f001 = Fz[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
            npos.z -= 2;
            f00_1 =Fz[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
        }
    break;
    }

    float3 grad(0.5f*(f100-f_100), 0.5f*(f010-f0_10), 0.5f*(f001-f00_1));


    return grad;
}

float sign(float a) {
    return a < 0 ? -1.0f: 1.0f;
}

float3 getTubeDirection(TubeSegmentation &T, int3 pos, int3 size) {

    // Do gradient on Fx, Fy and Fz and normalization
    float3 Fx = gradient(T, pos,0,1,size);
    float3 Fy = gradient(T, pos,1,2,size);
    float3 Fz = gradient(T, pos,2,3,size);

    float Hessian[3][3] = {
        {Fx.x, Fy.x, Fz.x},
        {Fy.x, Fy.y, Fz.y},
        {Fz.x, Fz.y, Fz.z}
    };
    float eigenValues[3];
    float eigenVectors[3][3];
    eigen_decomposition(Hessian, eigenVectors, eigenValues);
    float3 e1(eigenVectors[0][0], eigenVectors[1][0], eigenVectors[2][0]);
    return e1;
}

void doEigen(TubeSegmentation &T, int3 pos, int3 size, float3 * lambda, float3 * e1, float3 * e2, float3 * e3) {

    // Do gradient on Fx, Fy and Fz and normalization
    float3 Fx = gradient(T, pos,0,1,size);
    float3 Fy = gradient(T, pos,1,2,size);
    float3 Fz = gradient(T, pos,2,3,size);

    float Hessian[3][3] = {
        {Fx.x, Fy.x, Fz.x},
        {Fy.x, Fy.y, Fz.y},
        {Fz.x, Fz.y, Fz.z}
    };
    float eigenValues[3];
    float eigenVectors[3][3];
    eigen_decomposition(Hessian, eigenVectors, eigenValues);
    e1->x = eigenVectors[0][0];
    e1->y = eigenVectors[1][0];
    e1->z = eigenVectors[2][0];
    e2->x = eigenVectors[0][1];
    e2->y = eigenVectors[1][1];
    e2->z = eigenVectors[2][1];
    e3->x = eigenVectors[0][2];
    e3->y = eigenVectors[1][2];
    e3->z = eigenVectors[2][2];
    lambda->x = eigenValues[0];
    lambda->y = eigenValues[1];
    lambda->z = eigenValues[2];
}


char * runRidgeTraversal(TubeSegmentation &T, SIPL::int3 size, paramList &parameters, std::stack<CenterlinePoint> centerlineStack) {

    float Thigh = getParam(parameters, "tdf-high"); // 0.6
    int Dmin = getParam(parameters, "min-distance");
    float Mlow = getParam(parameters, "m-low"); // 0.2
    float Tlow = getParam(parameters, "tdf-low"); // 0.4
    int maxBelowTlow = getParam(parameters, "max-below-tdf-low"); // 2
    float minMeanTube = getParam(parameters, "min-mean-tdf"); //0.6
    int TreeMin = getParam(parameters, "min-tree-length"); // 200
    const int totalSize = size.x*size.y*size.z;

    int * centerlines = new int[totalSize]();
    INIT_TIMER

    // Create queue
    std::priority_queue<point, std::vector<point>, PointComparison> queue;

    START_TIMER
    // Collect all valid start points
    #pragma omp parallel for
    for(int z = 2; z < size.z-2; z++) {
        for(int y = 2; y < size.y-2; y++) {
            for(int x = 2; x < size.x-2; x++) {
                if(T.TDF[LPOS(x,y,z)] < Thigh)
                    continue;

                int3 pos(x,y,z);
                bool valid = true;
                for(int a = -1; a < 2; a++) {
                    for(int b = -1; b < 2; b++) {
                        for(int c = -1; c < 2; c++) {
                            int3 nPos(x+a,y+b,z+c);
                            if(SQR_MAG(nPos) < SQR_MAG(pos)) {
                                valid = false;
                                break;
                            }
                        }
                    }
                }

                if(valid) {
                    point p;
                    p.value = T.TDF[LPOS(x,y,z)];
                    p.x = x;
                    p.y = y;
                    p.z = z;
                    #pragma omp critical
                    queue.push(p);
                }
            }
        }
    }

    std::cout << "Processing " << queue.size() << " valid start points" << std::endl;
    if(queue.size() == 0) {
    	throw SIPL::SIPLException("no valid start points found", __LINE__, __FILE__);
    }
    STOP_TIMER("finding start points")
    START_TIMER
    int counter = 1;
    T.TDF[0] = 0;
    T.Fx[0] = 1;
    T.Fy[0] = 0;
    T.Fz[0] = 0;


    // Create a map of centerline distances
    unordered_map<int, int> centerlineDistances;

    // Create a map of centerline stacks
    unordered_map<int, std::stack<CenterlinePoint> > centerlineStacks;

    while(!queue.empty()) {
        // Traverse from new start point
        point p = queue.top();
        queue.pop();

        // Has it been handled before?
        if(centerlines[LPOS(p.x,p.y,p.z)] == 1)
            continue;

        unordered_set<int> newCenterlines;
        newCenterlines.insert(LPOS(p.x,p.y,p.z));
        int distance = 1;
        int connections = 0;
        int prevConnection = -1;
        int secondConnection = -1;
        float meanTube = T.TDF[LPOS(p.x,p.y,p.z)];

        // Create new stack for this centerline
        std::stack<CenterlinePoint> stack;
        CenterlinePoint startPoint;
        startPoint.pos.x = p.x;
        startPoint.pos.y = p.y;
        startPoint.pos.z = p.z;

        stack.push(startPoint);

        // For each direction
        for(int direction = -1; direction < 3; direction += 2) {
            int belowTlow = 0;
            int3 position(p.x,p.y,p.z);
            float3 t_i = getTubeDirection(T, position, size);
            t_i.x *= direction;
            t_i.y *= direction;
            t_i.z *= direction;
            float3 t_i_1;
            t_i_1.x = t_i.x;
            t_i_1.y = t_i.y;
            t_i_1.z = t_i.z;


            // Traverse
            while(true) {
                int3 maxPoint(0,0,0);

                // Check for out of bounds
                if(position.x < 3 || position.x > size.x-3 || position.y < 3 || position.y > size.y-3 || position.z < 3 || position.z > size.z-3)
                    break;

                // Try to find next point from all neighbors
                for(int a = -1; a < 2; a++) {
                    for(int b = -1; b < 2; b++) {
                        for(int c = -1; c < 2; c++) {
                            int3 n(position.x+a,position.y+b,position.z+c);
                            if((a == 0 && b == 0 && c == 0) || T.TDF[POS(n)] == 0.0f)
                                continue;

                            float3 dir((float)(n.x-position.x),(float)(n.y-position.y),(float)(n.z-position.z));
                            dir = normalize(dir);
                            if( (dir.x*t_i.x+dir.y*t_i.y+dir.z*t_i.z) <= 0.1)
                                continue;

                            if(T.radius[POS(n)] >= 1.5f) {
                                if(M(n.x,n.y,n.z) > M(maxPoint.x,maxPoint.y,maxPoint.z))
                                maxPoint = n;
                            } else {
                                if(T.TDF[LPOS(n.x,n.y,n.z)]*M(n.x,n.y,n.z) > T.TDF[POS(maxPoint)]*M(maxPoint.x,maxPoint.y,maxPoint.z))
                                maxPoint = n;
                            }

                        }
                    }
                }

                if(maxPoint.x+maxPoint.y+maxPoint.z > 0) {
                    // New maxpoint found, check it!
                    if(centerlines[LPOS(maxPoint.x,maxPoint.y,maxPoint.z)] > 0) {
                        // Hit an existing centerline
                        if(prevConnection == -1) {
                            prevConnection = centerlines[LPOS(maxPoint.x,maxPoint.y,maxPoint.z)];
                        } else {
                            if(prevConnection ==centerlines[LPOS(maxPoint.x,maxPoint.y,maxPoint.z)]) {
                                // A loop has occured, reject this centerline
                                connections = 5;
                            } else {
                                secondConnection = centerlines[LPOS(maxPoint.x,maxPoint.y,maxPoint.z)];
                            }
                        }
                        break;
                    } else if(M(maxPoint.x,maxPoint.y,maxPoint.z) < Mlow || (belowTlow > maxBelowTlow && T.TDF[LPOS(maxPoint.x,maxPoint.y,maxPoint.z)] < Tlow)) {
                        // New point is below thresholds
                        break;
                    } else if(newCenterlines.count(LPOS(maxPoint.x,maxPoint.y,maxPoint.z)) > 0) {
                        // Loop detected!
                        break;
                    } else {
                        // Point is OK, proceed to add it and continue
                        if(T.TDF[LPOS(maxPoint.x,maxPoint.y,maxPoint.z)] < Tlow) {
                            belowTlow++;
                        } else {
                            belowTlow = 0;
                        }

                        // Update direction
                        //float3 e1 = getTubeDirection(T, maxPoint,size.x,size.y,size.z);

                        //TODO: check if all eigenvalues are negative, if so find the egeinvector that best matches
                        float3 lambda, e1, e2, e3;
                        doEigen(T, maxPoint, size, &lambda, &e1, &e2, &e3);
                        if((lambda.x < 0 && lambda.y < 0 && lambda.z < 0)) {
                            if(fabs(dot(t_i, e3)) > fabs(dot(t_i, e2))) {
                                if(fabs(dot(t_i, e3)) > fabs(dot(t_i, e1))) {
                                    e1 = e3;
                                }
                            } else if(fabs(dot(t_i, e2)) > fabs(dot(t_i, e1))) {
                                e1 = e2;
                            }
                        }


                        float maintain_dir = sign(dot(e1,t_i));
                        float3 vec_sum;
                        vec_sum.x = maintain_dir*e1.x + t_i.x + t_i_1.x;
                        vec_sum.y = maintain_dir*e1.y + t_i.y + t_i_1.y;
                        vec_sum.z = maintain_dir*e1.z + t_i.z + t_i_1.z;
                        vec_sum = normalize(vec_sum);
                        t_i_1 = t_i;
                        t_i = vec_sum;

                        // update position
                        position = maxPoint;
                        distance ++;
                        newCenterlines.insert(LPOS(maxPoint.x,maxPoint.y,maxPoint.z));
                        meanTube += T.TDF[LPOS(maxPoint.x,maxPoint.y,maxPoint.z)];

                        // Create centerline point
                        CenterlinePoint p;
                        p.pos = position;
                        p.next = &(stack.top()); // add previous
                        if(T.radius[POS(p.pos)] > 3.0f) {
                            p.large = true;
                        } else {
                            p.large = false;
                        }

                        // Add point to stack
                        stack.push(p);
                    }
                } else {
                    // No maxpoint found, stop!
                    break;
                }

            } // End traversal
        } // End for each direction

        // Check to see if new traversal can be added
        //std::cout << "Finished. Distance " << distance << " meanTube: " << meanTube/distance << std::endl;
        if(distance > Dmin && meanTube/distance > minMeanTube && connections < 2) {
            //std::cout << "Finished. Distance " << distance << " meanTube: " << meanTube/distance << std::endl;
            //std::cout << "------------------- New centerlines added #" << counter << " -------------------------" << std::endl;

            unordered_set<int>::iterator usit;
            if(prevConnection == -1) {
                // No connections
                for(usit = newCenterlines.begin(); usit != newCenterlines.end(); usit++) {
                    centerlines[*usit] = counter;
                }
                centerlineDistances[counter] = distance;
                centerlineStacks[counter] = stack;
                counter ++;
            } else {
                // The first connection

                std::stack<CenterlinePoint> prevConnectionStack = centerlineStacks[prevConnection];
                while(!stack.empty()) {
                    prevConnectionStack.push(stack.top());
                    stack.pop();
                }

                for(usit = newCenterlines.begin(); usit != newCenterlines.end(); usit++) {
                    centerlines[*usit] = prevConnection;
                }
                centerlineDistances[prevConnection] += distance;
                if(secondConnection != -1) {
                    // Two connections, move secondConnection to prevConnection
                    std::stack<CenterlinePoint> secondConnectionStack = centerlineStacks[secondConnection];
                    centerlineStacks.erase(secondConnection);
                    while(!secondConnectionStack.empty()) {
                        prevConnectionStack.push(secondConnectionStack.top());
                        secondConnectionStack.pop();
                    }

                    #pragma omp parallel for
                    for(int i = 0; i < totalSize;i++) {
                        if(centerlines[i] == secondConnection)
                            centerlines[i] = prevConnection;
                    }
                    centerlineDistances[prevConnection] += centerlineDistances[secondConnection];
                    centerlineDistances.erase(secondConnection);
                }

                centerlineStacks[prevConnection] = prevConnectionStack;
            }
        } // end if new point can be added
    } // End while queue is not empty
    std::cout << "Finished traversal" << std::endl;
    STOP_TIMER("traversal")
    START_TIMER

    if(centerlineDistances.size() == 0) {
        //throw SIPL::SIPLException("no centerlines were extracted");
        char * returnCenterlines = new char[totalSize]();
        return returnCenterlines;
    }

    // Find largest connected tree and all trees above a certain size
    unordered_map<int, int>::iterator it;
    int max = centerlineDistances.begin()->first;
    std::list<int> trees;
    for(it = centerlineDistances.begin(); it != centerlineDistances.end(); it++) {
        if(it->second > centerlineDistances[max])
            max = it->first;
        if(it->second > TreeMin)
            trees.push_back(it->first);
    }
    std::list<int>::iterator it2;
    // TODO: if use the method with TreeMin have to add them to centerlineStack also
    centerlineStack = centerlineStacks[max];
    for(it2 = trees.begin(); it2 != trees.end(); it2++) {
        while(!centerlineStacks[*it2].empty()) {
            centerlineStack.push(centerlineStacks[*it2].top());
            centerlineStacks[*it2].pop();
        }
    }

    char * returnCenterlines = new char[totalSize]();
    // Mark largest tree with 1, and rest with 0
    #pragma omp parallel for
    for(int i = 0; i < totalSize;i++) {
        if(centerlines[i] == max) {
        //if(centerlines[i] > 0) {
            returnCenterlines[i] = 1;
        } else {
            bool valid = false;
            for(it2 = trees.begin(); it2 != trees.end(); it2++) {
                if(centerlines[i] == *it2) {
                    returnCenterlines[i] = 1;
                    valid = true;
                    break;
                }
            }
            if(!valid)
                returnCenterlines[i] = 0;

        }
    }
    STOP_TIMER("finding largest tree")

	delete[] centerlines;
    return returnCenterlines;
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

Image3D runFastGVF(OpenCL &ocl, Image3D *vectorField, paramList &parameters, SIPL::int3 &size) {

    const int GVFIterations = getParam(parameters, "gvf-iterations");
    const bool no3Dwrite = !getParamBool(parameters, "3d_write");
    const float MU = getParam(parameters, "gvf-mu");
    const int totalSize = size.x*size.y*size.z;

    Kernel GVFInitKernel = Kernel(ocl.program, "GVF3DInit");
    Kernel GVFIterationKernel = Kernel(ocl.program, "GVF3DIteration");
    Kernel GVFFinishKernel = Kernel(ocl.program, "GVF3DFinish");
    Image3D resultVectorField;

    std::cout << "Running GVF with " << GVFIterations << " iterations " << std::endl;
    if(no3Dwrite) {
    	int vectorFieldSize = sizeof(float);
    	if(getParamBool(parameters, "16bit-vectors"))
    		vectorFieldSize = sizeof(short);
        // Create auxillary buffers
        Buffer * vectorFieldBuffer = new Buffer(
                ocl.context,
                CL_MEM_READ_WRITE,
                3*vectorFieldSize*totalSize
        );
        GC->addMemObject(vectorFieldBuffer);
        Buffer * vectorFieldBuffer1 = new Buffer(
                ocl.context,
                CL_MEM_READ_WRITE,
                3*vectorFieldSize*totalSize
        );
        GC->addMemObject(vectorFieldBuffer1);

        GVFInitKernel.setArg(0, *vectorField);
        GVFInitKernel.setArg(1, *vectorFieldBuffer);
        ocl.queue.enqueueNDRangeKernel(
                GVFInitKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NullRange
        );

        // Run iterations
        GVFIterationKernel.setArg(0, *vectorField);
        GVFIterationKernel.setArg(3, MU);

        for(int i = 0; i < GVFIterations; i++) {
            if(i % 2 == 0) {
                GVFIterationKernel.setArg(1, *vectorFieldBuffer);
                GVFIterationKernel.setArg(2, *vectorFieldBuffer1);
            } else {
                GVFIterationKernel.setArg(1, *vectorFieldBuffer1);
                GVFIterationKernel.setArg(2, *vectorFieldBuffer);
            }
                ocl.queue.enqueueNDRangeKernel(
                        GVFIterationKernel,
                        NullRange,
                        NDRange(size.x,size.y,size.z),
                        NDRange(4,4,4)
                );
        }
        ocl.queue.finish(); //This finish is necessary
        GC->deleteMemObject(vectorFieldBuffer1);
        GC->deleteMemObject(vectorField);

        Buffer finalVectorFieldBuffer = Buffer(
                ocl.context,
                CL_MEM_WRITE_ONLY,
                4*vectorFieldSize*totalSize
        );

        // Copy vector field to image
        GVFFinishKernel.setArg(0, *vectorFieldBuffer);
        GVFFinishKernel.setArg(1, finalVectorFieldBuffer);

        ocl.queue.enqueueNDRangeKernel(
                GVFFinishKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NDRange(4,4,4)
        );
        ocl.queue.finish();
        GC->deleteMemObject(vectorFieldBuffer);

		cl::size_t<3> offset;
		offset[0] = 0;
		offset[1] = 0;
		offset[2] = 0;
		cl::size_t<3> region;
		region[0] = size.x;
		region[1] = size.y;
		region[2] = size.z;

        // Copy buffer contents to image
		if(getParamBool(parameters, "16bit-vectors")) {
            resultVectorField = Image3D(ocl.context, CL_MEM_READ_ONLY, ImageFormat(CL_RGBA, CL_SNORM_INT16), size.x, size.y, size.z);
        } else {
            resultVectorField = Image3D(ocl.context, CL_MEM_READ_ONLY, ImageFormat(CL_RGBA, CL_FLOAT), size.x, size.y, size.z);
        }
        ocl.queue.enqueueCopyBufferToImage(
                finalVectorFieldBuffer,
                resultVectorField,
                0,
                offset,
                region
        );

    } else {
        Image3D vectorField1;
        Image3D initVectorField;
        if(getParamBool(parameters, "16bit-vectors")) {
            vectorField1 = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_RGBA, CL_SNORM_INT16), size.x, size.y, size.z);
            initVectorField = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_RG, CL_SNORM_INT16), size.x, size.y, size.z);
        } else {
            vectorField1 = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_RGBA, CL_FLOAT), size.x, size.y, size.z);
            initVectorField = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_RG, CL_FLOAT), size.x, size.y, size.z);
        }

        // init vectorField from image
        GVFInitKernel.setArg(0, *vectorField);
        GVFInitKernel.setArg(1, vectorField1);
        GVFInitKernel.setArg(2, initVectorField);
        ocl.queue.enqueueNDRangeKernel(
                GVFInitKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NDRange(4,4,4)
        );
        // Run iterations
        GVFIterationKernel.setArg(0, initVectorField);
        GVFIterationKernel.setArg(3, MU);

        for(int i = 0; i < GVFIterations; i++) {
            if(i % 2 == 0) {
                GVFIterationKernel.setArg(1, vectorField1);
                GVFIterationKernel.setArg(2, *vectorField);
            } else {
                GVFIterationKernel.setArg(1, *vectorField);
                GVFIterationKernel.setArg(2, vectorField1);
            }
                ocl.queue.enqueueNDRangeKernel(
                        GVFIterationKernel,
                        NullRange,
                        NDRange(size.x,size.y,size.z),
                        NDRange(4,4,4)
                );
        }
        ocl.queue.finish();
        GC->deleteMemObject(vectorField);

        // Copy vector field to image
		if(getParamBool(parameters, "16bit-vectors")) {
            resultVectorField = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_RGBA, CL_SNORM_INT16), size.x, size.y, size.z);
        } else {
            resultVectorField = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_RGBA, CL_FLOAT), size.x, size.y, size.z);
        }
        GVFFinishKernel.setArg(0, vectorField1);
        GVFFinishKernel.setArg(1, resultVectorField);

        ocl.queue.enqueueNDRangeKernel(
                GVFFinishKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NDRange(4,4,4)
        );
    }
    return resultVectorField;
}
Image3D runLowMemoryGVF(OpenCL &ocl, Image3D * vectorField, paramList &parameters, SIPL::int3 &size) {

    const int GVFIterations = getParam(parameters, "gvf-iterations");
    const bool no3Dwrite = !getParamBool(parameters, "3d_write");
    const float MU = getParam(parameters, "gvf-mu");
    const int totalSize = size.x*size.y*size.z;

    Kernel GVFInitKernel = Kernel(ocl.program, "GVF3DInit_one_component");
    Kernel GVFIterationKernel = Kernel(ocl.program, "GVF3DIteration_one_component");
    Kernel GVFFinishKernel = Kernel(ocl.program, "GVF3DFinish_one_component");

    Image3D resultVectorField;
    std::cout << "Running GVF with " << GVFIterations << " iterations " << std::endl;
    if(no3Dwrite) {
    	int vectorFieldSize = sizeof(float);
    	if(getParamBool(parameters, "16bit-vectors")) {
    		vectorFieldSize = sizeof(short);
    	}
    	Buffer *vectorFieldX;
    	Buffer *vectorFieldY;
    	Buffer *vectorFieldZ;
        for(int component = 1; component < 4; component++) {

        	Buffer * vectorField1 = new Buffer(
                ocl.context,
                CL_MEM_READ_WRITE,
                vectorFieldSize*totalSize
			);
            GC->addMemObject(vectorField1);
			Buffer initVectorField = Buffer(
                ocl.context,
                CL_MEM_READ_WRITE,
                2*vectorFieldSize*totalSize
			);

			GVFInitKernel.setArg(0, *vectorField);
			GVFInitKernel.setArg(1, *vectorField1);
			GVFInitKernel.setArg(2, initVectorField);
			GVFInitKernel.setArg(3, component);
			ocl.queue.enqueueNDRangeKernel(
					GVFInitKernel,
					NullRange,
					NDRange(size.x,size.y,size.z),
					NullRange
			);
			ocl.queue.finish();

			Buffer vectorField2 = Buffer(
                ocl.context,
                CL_MEM_READ_WRITE,
                vectorFieldSize*totalSize
			);

			// Run iterations
			GVFIterationKernel.setArg(0, initVectorField);
			GVFIterationKernel.setArg(3, MU);

			for(int i = 0; i < GVFIterations; i++) {
				if(i % 2 == 0) {
					GVFIterationKernel.setArg(1, *vectorField1);
					GVFIterationKernel.setArg(2, vectorField2);
				} else {
					GVFIterationKernel.setArg(1, vectorField2);
					GVFIterationKernel.setArg(2, *vectorField1);
				}
					ocl.queue.enqueueNDRangeKernel(
							GVFIterationKernel,
							NullRange,
							NDRange(size.x,size.y,size.z),
							NullRange
					);
			}
			if(component == 1) {
				vectorFieldX = vectorField1;
			} else if(component == 2) {
				vectorFieldY = vectorField1;
			} else {
				vectorFieldZ = vectorField1;
			}
			ocl.queue.finish();
			std::cout << "finished component " << component << std::endl;
        }
        GC->deleteMemObject(vectorField);


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

        // Copy vector field to image
        GVFFinishKernel.setArg(0, *vectorFieldX);
        GVFFinishKernel.setArg(1, *vectorFieldY);
        GVFFinishKernel.setArg(2, *vectorFieldZ);
        GVFFinishKernel.setArg(3, vectorFieldBuffer);
        GVFFinishKernel.setArg(4, vectorFieldBuffer2);
        GVFFinishKernel.setArg(5, maxZ);

        ocl.queue.enqueueNDRangeKernel(
                GVFFinishKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NullRange
        );

        ocl.queue.finish();
        GC->deleteMemObject(vectorFieldX);
        GC->deleteMemObject(vectorFieldY);
        GC->deleteMemObject(vectorFieldZ);

		cl::size_t<3> offset;
		offset[0] = 0;
		offset[1] = 0;
		offset[2] = 0;
		cl::size_t<3> region;
		region[0] = size.x;
		region[1] = size.y;
		region[2] = size.z;

		if(getParamBool(parameters, "16bit-vectors")) {
            resultVectorField = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_RGBA, CL_SNORM_INT16), size.x, size.y, size.z);
        } else {
            resultVectorField = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_RGBA, CL_FLOAT), size.x, size.y, size.z);
        }
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
					resultVectorField,
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
					resultVectorField,
					0,
					offset2,
					region3
			);
		} else {
			// Copy buffer contents to image
			ocl.queue.enqueueCopyBufferToImage(
					vectorFieldBuffer,
					resultVectorField,
					0,
					offset,
					region
			);
		}

    } else {
        Image3D vectorFieldX, vectorFieldY, vectorFieldZ;
        for(int component = 1; component < 4; component++) {
        	Image3D initVectorField, vectorField1, vectorField2;
        	if(getParamBool(parameters, "32bit-vectors")) {
				vectorField1 = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_R, CL_FLOAT), size.x, size.y, size.z);
				vectorField2 = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_R, CL_FLOAT), size.x, size.y, size.z);
				initVectorField = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_RG, CL_FLOAT), size.x, size.y, size.z);
			} else {
				vectorField1 = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_R, CL_SNORM_INT16), size.x, size.y, size.z);
				vectorField2 = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_R, CL_SNORM_INT16), size.x, size.y, size.z);
				initVectorField = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_RG, CL_SNORM_INT16), size.x, size.y, size.z);
			}

			// init vectorField from image
			GVFInitKernel.setArg(0, vectorField);
			GVFInitKernel.setArg(1, vectorField1);
			GVFInitKernel.setArg(2, initVectorField);
			GVFInitKernel.setArg(3, component);
			ocl.queue.enqueueNDRangeKernel(
					GVFInitKernel,
					NullRange,
					NDRange(size.x,size.y,size.z),
					NDRange(4,4,4)
			);

			// Run iterations
			GVFIterationKernel.setArg(0, initVectorField);
			GVFIterationKernel.setArg(3, MU);

			for(int i = 0; i < GVFIterations; i++) {
				if(i % 2 == 0) {
					GVFIterationKernel.setArg(1, vectorField1);
					GVFIterationKernel.setArg(2, vectorField2);
				} else {
					GVFIterationKernel.setArg(1, vectorField2);
					GVFIterationKernel.setArg(2, vectorField1);
				}
				ocl.queue.enqueueNDRangeKernel(
					GVFIterationKernel,
					NullRange,
					NDRange(size.x,size.y,size.z),
					NDRange(4,4,4)
				);
			}
			if(component == 1) {
				vectorFieldX = vectorField1;
			} else if(component == 2) {
				vectorFieldY = vectorField1;
			} else {
				vectorFieldZ = vectorField1;
			}
			ocl.queue.finish();
			std::cout << "finished component " << component << std::endl;
        }

		if(getParamBool(parameters, "16bit-vectors")) {
            resultVectorField = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_RGBA, CL_SNORM_INT16), size.x, size.y, size.z);
        } else {
            resultVectorField = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_RGBA, CL_FLOAT), size.x, size.y, size.z);
        }
        // Copy vector fields to image
        GVFFinishKernel.setArg(0, vectorFieldX);
        GVFFinishKernel.setArg(1, vectorFieldY);
        GVFFinishKernel.setArg(2, vectorFieldZ);
        GVFFinishKernel.setArg(3, resultVectorField);

        ocl.queue.enqueueNDRangeKernel(
                GVFFinishKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NDRange(4,4,4)
        );
    }

    return resultVectorField;
}


Image3D runGVF(OpenCL &ocl, Image3D * vectorField, paramList &parameters, SIPL::int3 &size, bool useLessMemory) {

	if(useLessMemory) {
		std::cout << "NOTE: Running slow GVF that uses less memory." << std::endl;
		return runLowMemoryGVF(ocl,vectorField,parameters,size);
	} else {
		std::cout << "NOTE: Running fast GVF." << std::endl;
		return runFastGVF(ocl,vectorField,parameters,size);
	}
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
    Kernel circleFittingTDFKernel(ocl.program, "circleFittingTDF");
    Kernel combineKernel = Kernel(ocl.program, "combine");

    cl::Event startEvent, endEvent;
    cl_ulong start, end;
    INIT_TIMER
    void * TDFsmall;
    float * radiusSmall;
    if(radiusMin < 2.5f) {
        Image3D * blurredVolume = new Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_R, CL_FLOAT), size.x, size.y, size.z);
        GC->addMemObject(blurredVolume);
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
            GC->deleteMemObject(blurredVolume);
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
        GC->addMemObject(vectorFieldSmall);
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
        GC->addMemObject(vectorFieldSmall);

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
        GC->deleteMemObject(blurredVolume);
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
    GC->addMemObject(TDFsmallBuffer);
    Buffer * radiusSmallBuffer = new Buffer(ocl.context, CL_MEM_WRITE_ONLY, sizeof(float)*totalSize);
    GC->addMemObject(radiusSmallBuffer);
    circleFittingTDFKernel.setArg(0, *vectorFieldSmall);
    circleFittingTDFKernel.setArg(1, *TDFsmallBuffer);
    circleFittingTDFKernel.setArg(2, *radiusSmallBuffer);
    circleFittingTDFKernel.setArg(3, radiusMin);
    circleFittingTDFKernel.setArg(4, 3.0f);
    circleFittingTDFKernel.setArg(5, 0.5f);

    ocl.queue.enqueueNDRangeKernel(
            circleFittingTDFKernel,
            NullRange,
            NDRange(size.x,size.y,size.z),
            NDRange(4,4,4)
    );


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
        GC->deleteMemObject(dataset);
		return;
    } else {
        ocl.queue.finish();
        GC->deleteMemObject(vectorFieldSmall);
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
    GC->deleteMemObject(TDFsmallBuffer);
    GC->deleteMemObject(radiusSmallBuffer);
    } // end if radiusMin < 2.5


if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&endEvent);
    ocl.queue.finish();
    startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    endEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &end);
    std::cout << "RUNTIME of TDF small: " << (end-start)*1.0e-6 << " ms" << std::endl;
}
    /* Large Airways */

if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&startEvent);
}
    Image3D * blurredVolume = new Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_R, CL_FLOAT), size.x, size.y, size.z);
    GC->addMemObject(blurredVolume);
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
        GC->deleteMemObject(dataset);
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
			GC->addMemObject(initVectorField);
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
			GC->addMemObject(initVectorField);
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
        GC->deleteMemObject(blurredVolume);

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
        GC->addMemObject(initVectorField);


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
        GC->deleteMemObject(blurredVolume);
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
	if(useSlowGVF) {
		vectorField = runGVF(ocl, initVectorField, parameters, size, true);
	} else {
		vectorField = runGVF(ocl, initVectorField, parameters, size, false);
	}

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

    circleFittingTDFKernel.setArg(0, vectorField);
    circleFittingTDFKernel.setArg(1, TDFlarge);
    circleFittingTDFKernel.setArg(2, radiusLarge);
    circleFittingTDFKernel.setArg(3, std::max(2.5f, radiusMin));
    circleFittingTDFKernel.setArg(4, radiusMax);
    circleFittingTDFKernel.setArg(5, 1.5f);

    ocl.queue.enqueueNDRangeKernel(
            circleFittingTDFKernel,
            NullRange,
            NDRange(size.x,size.y,size.z),
            NDRange(4,4,4)
    );

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

class Edge;
class Node {
public:
	int3 pos;
	std::vector<Edge *> edges;

};

class Edge {
public:
	Node * source;
	Node * target;
	std::vector<Node *> removedNodes;
	float distance;
};

class EdgeComparator {
public:
	bool operator()(Edge * a, Edge *b) const {
		return a->distance > b->distance;
	}
};


void removeEdge(Node * n, Node * remove) {
	std::vector<Edge *> edges;
	for(int i = 0; i < n->edges.size(); i++) {
		Edge * e = n->edges[i];
		if(e->target != remove) {
			edges.push_back(e);
		}
	}
	n->edges = edges;
}

void restoreNodes(Edge * e, std::vector<Node *> &finalGraph) {
	if(e->removedNodes.size() == 0)
		return;

	// Remove e from e->source
	removeEdge(e->source, e->target);
	Node * previous = e->source;
	for(int k = 0; k < e->removedNodes.size(); k++) {
		Edge * newEdge = new Edge;
		// Create edge from source to k or k-1 to k
		newEdge->source = previous;
		newEdge->target = e->removedNodes[k];
		previous->edges.push_back(newEdge);
		previous = e->removedNodes[k];
		finalGraph.push_back(e->removedNodes[k]);

	}
	// Create edge from last k to target
	Edge * newEdge = new Edge;
	newEdge->source = previous;
	newEdge->target = e->target;
	previous->edges.push_back(newEdge);
}

std::vector<Node *> minimumSpanningTreePCE(
		int root,
		std::vector<Node *> &graph,
		int3 &size,
		unordered_set<int> &visited
		) {
	std::vector<Node *> result;
	std::priority_queue<Edge *, std::vector<Edge *>, EdgeComparator> queue;

	// Start with graph[0]
	result.push_back(graph[root]);
	visited.insert(POS(graph[root]->pos));

	// Add edges to priority queue
	for(int i = 0; i < graph[root]->edges.size(); i++) {
		Edge * en = graph[root]->edges[i];
		queue.push(en);
	}
	graph[root]->edges.clear();

	while(!queue.empty()) {
		Edge * e = queue.top();
		queue.pop();

		if(visited.find(POS(e->target->pos)) != visited.end())
			continue; // already visited

		// Add all edges of e->target to queue if targets have not been added
		for(int i = 0; i < e->target->edges.size(); i++) {
			Edge * en = e->target->edges[i];
			if(visited.find(POS(en->target->pos)) == visited.end()) {
				queue.push(en);
			}
		}

		e->target->edges.clear(); // Remove all edges first
		e->source->edges.push_back(e); // Add edge to source
		result.push_back(e->target); // Add target to result
		visited.insert(POS(e->target->pos));
	}

	return result;
}

void removeLoops(
		std::vector<int3> &vertices,
		std::vector<SIPL::int2> &edges,
		int3 &size
	) {

	std::vector<Node *> nodes;
	for(int i = 0; i < vertices.size(); i++) {
		Node * n = new Node;
		n->pos = vertices[i];
		nodes.push_back(n);
	}
	for(int i = 0; i < edges.size(); i++) {
		Node * a = nodes[edges[i].x];
		Node * b = nodes[edges[i].y];
		Edge * e = new Edge;
		e->distance = a->pos.distance(b->pos);
		e->source = a;
		e->target = b;
		a->edges.push_back(e);

		Edge * e2 = new Edge;
		e2->distance = a->pos.distance(b->pos);
		e2->source = b;
		e2->target = a;
		b->edges.push_back(e2);
	}

	// Create graph
	std::vector<Node *> graph;

	// Remove all nodes with degree 2 and add them to edge
	for(int i = 0; i < nodes.size(); i++) {
		Node * n = nodes[i];
		if(n->edges.size() == 2) {
			// calculate distance
			float distance = n->edges[0]->distance+n->edges[1]->distance;
			Node * a = n->edges[0]->target;
			Node * b = n->edges[1]->target;
			// Fuse the two nodes together
			Edge * e = new Edge;
			e->source = a;
			e->target = b;
			e->distance = distance;
			// Add removed nodes from previous edges
			for(int j = n->edges[0]->removedNodes.size()-1; j >= 0; j--) {
				e->removedNodes.push_back(n->edges[0]->removedNodes[j]);
			}
			e->removedNodes.push_back(n);
			for(int j = 0; j < n->edges[1]->removedNodes.size(); j++) {
				e->removedNodes.push_back(n->edges[1]->removedNodes[j]);
			}

			Edge * e2 = new Edge;
			e2->source = b;
			e2->target = a;
			e2->distance = distance;
			// Add removed nodes from previous edges
			for(int j = n->edges[1]->removedNodes.size()-1; j >= 0; j--) {
				e2->removedNodes.push_back(n->edges[1]->removedNodes[j]);
			}
			e2->removedNodes.push_back(n);
			for(int j = 0; j < n->edges[0]->removedNodes.size(); j++) {
				e2->removedNodes.push_back(n->edges[0]->removedNodes[j]);
			}

			removeEdge(a,n);
			removeEdge(b,n);
			n->edges.clear();
			a->edges.push_back(e);
			b->edges.push_back(e2);
		} else {
			// add node to graph
			graph.push_back(n);
		}
	}

	// Do MST with edge distance as cost
	int sizeBeforeMST = graph.size();
	if(sizeBeforeMST == 0) {
	    throw SIPL::SIPLException("Centerline graph size is 0! Can't continue. Maybe lower min-tree-length?", __LINE__,__FILE__);
	}
	unordered_set<int> visited;
	std::vector<Node *> newGraph = minimumSpanningTreePCE(0, graph, size, visited);
	int sizeAfterMST = newGraph.size();

	while(newGraph.size() < graph.size()) {
		// Some nodes were not visited: find new root
		int root;
		for(int i = 0; i < graph.size(); i++) {
			if(visited.find(POS(graph[i]->pos)) == visited.end()) {
				// i has not been used
				root = i;
			}
		}

		std::vector<Node *> newGraph2 = minimumSpanningTreePCE(root, graph, size, visited);
		for(int i = 0; i < newGraph2.size(); i++) {
			newGraph.push_back(newGraph2[i]);
		}
		int sizeAfterMST = newGraph.size();
	}

	// Restore graph
	// For all edges that are in the MST graph: get nodes that was on these edges
	std::vector<Node *> finalGraph;
	for(int i = 0; i < newGraph.size(); i++) {
		Node * n = newGraph[i];
		finalGraph.push_back(n);
		std::vector<Edge *> nEdges = n->edges;
		for(int j = 0; j < nEdges.size(); j++) {
			Edge * e = nEdges[j];
			restoreNodes(e, finalGraph);
		}
	}


	std::vector<int3> newVertices;
	std::vector<SIPL::int2> newEdges;
	unordered_map<int, int> added;
	int counter = 0;
	for(int i = 0; i < finalGraph.size(); i++) {
		Node * n = finalGraph[i];
		int nIndex;
		if(added.find(POS(n->pos)) != added.end()) {
			// already has been added
			// fetch index
			nIndex = added[POS(n->pos)];
		} else {
			newVertices.push_back(n->pos);
			added[POS(n->pos)] = counter;
			nIndex = counter;
			counter++;
		}
		for(int j = 0; j < n->edges.size(); j++) {
			Edge * e = n->edges[j];
			// add edge from n to n_j

			int tIndex;
			if(added.find(POS(e->target->pos)) != added.end()) {
				// already has been added
				// fetch index
				tIndex = added[POS(e->target->pos)];
			} else {
				newVertices.push_back(e->target->pos);
				added[POS(e->target->pos)] = counter;
				tIndex = counter;
				counter++;
			}
			newEdges.push_back(SIPL::int2(nIndex, tIndex));
		}
	}

	// Cleanup
	vertices = newVertices;
	edges = newEdges;
}

char * createCenterlineVoxels(
		std::vector<int3> &vertices,
		std::vector<SIPL::int2> &edges,
		float * radius,
		int3 &size
		) {
	const int totalSize = size.x*size.y*size.z;
	char * centerlines = new char[totalSize]();

	for(int i = 0; i < edges.size(); i++) {
		int3 a = vertices[edges[i].x];
		int3 b = vertices[edges[i].y];
		float distance = a.distance(b);
		float3 direction(b.x-a.x,b.y-a.y,b.z-a.z);
		int n = ceil(distance);
		float avgRadius = 0.0f;
		for(int j = 0; j < n; j++) {
			float ratio = (float)j/n;
			float3 pos = a+direction*ratio;
			int3 intPos(round(pos.x), round(pos.y), round(pos.z));
			centerlines[POS(intPos)] = 1;
			avgRadius += radius[POS(intPos)];
		}
		avgRadius /= n;

		for(int j = 0; j < n; j++) {
			float ratio = (float)j/n;
			float3 pos = a+direction*ratio;
			int3 intPos(round(pos.x), round(pos.y), round(pos.z));
			radius[POS(intPos)] = avgRadius;
		}

	}

	return centerlines;
}

Image3D runNewCenterlineAlgWithoutOpenCL(OpenCL &ocl, SIPL::int3 size, paramList &parameters, Image3D &vectorField, Image3D &TDF, Image3D &radius) {
    const int totalSize = size.x*size.y*size.z;
	const bool no3Dwrite = !getParamBool(parameters, "3d_write");
    const int cubeSize = getParam(parameters, "cube-size");
    const int minTreeLength = getParam(parameters, "min-tree-length");
    const float Thigh = getParam(parameters, "tdf-high");
    const float minAvgTDF = getParam(parameters, "min-mean-tdf");
    const float maxDistance = getParam(parameters, "max-distance");

    cl::size_t<3> offset;
    offset[0] = 0;
    offset[1] = 0;
    offset[2] = 0;
    cl::size_t<3> region;
    region[0] = size.x;
    region[1] = size.y;
    region[2] = size.z;

    // Transfer TDF, vectorField and radius to host
    TubeSegmentation T;
    T.Fx = new float[totalSize];
    T.Fy = new float[totalSize];
    T.Fz = new float[totalSize];
    T.TDF = new float[totalSize];

    if(!getParamBool(parameters, "16bit-vectors")) {
    	// 32 bit vector fields
        float * Fs = new float[totalSize*4];
        ocl.queue.enqueueReadImage(vectorField, CL_TRUE, offset, region, 0, 0, Fs);
#pragma omp parallel for
        for(int i = 0; i < totalSize; i++) {
            T.Fx[i] = Fs[i*4];
            T.Fy[i] = Fs[i*4+1];
            T.Fz[i] = Fs[i*4+2];
        }
        delete[] Fs;
        ocl.queue.enqueueReadImage(TDF, CL_TRUE, offset, region, 0, 0, T.TDF);
    } else {
    	// 16 bit vector fields
        short * Fs = new short[totalSize*4];
        ocl.queue.enqueueReadImage(vectorField, CL_TRUE, offset, region, 0, 0, Fs);
#pragma omp parallel for
        for(int i = 0; i < totalSize; i++) {
            T.Fx[i] = MAX(-1.0f, Fs[i*4] / 32767.0f);
            T.Fy[i] = MAX(-1.0f, Fs[i*4+1] / 32767.0f);;
            T.Fz[i] = MAX(-1.0f, Fs[i*4+2] / 32767.0f);
        }
        delete[] Fs;

        // Convert 16 bit TDF to 32 bit
        unsigned short * tempTDF = new unsigned short[totalSize];
        ocl.queue.enqueueReadImage(TDF, CL_TRUE, offset, region, 0, 0, tempTDF);
#pragma omp parallel for
        for(int i = 0; i < totalSize; i++) {
            T.TDF[i] = (float)tempTDF[i] / 65535.0f;
        }
        delete[] tempTDF;
    }
    T.radius = new float[totalSize];
    ocl.queue.enqueueReadImage(radius, CL_TRUE, offset, region, 0, 0, T.radius);

    // Get candidate points
    std::vector<int3> candidatePoints;
#pragma omp parallel for
    for(int z = 1; z < size.z-1; z++) {
    for(int y = 1; y < size.y-1; y++) {
    for(int x = 1; x < size.x-1; x++) {
       int3 pos(x,y,z);
       if(T.TDF[POS(pos)] >= Thigh) {
#pragma omp critical
           candidatePoints.push_back(pos);
       }
    }}}
    std::cout << "candidate points: " << candidatePoints.size() << std::endl;

    unordered_set<int> filteredPoints;
#pragma omp parallel for
    for(int i = 0; i < candidatePoints.size(); i++) {
        int3 pos = candidatePoints[i];
        // Filter candidate points
        const float thetaLimit = 0.5f;
        const float radii = T.radius[POS(pos)];
        const int maxD = std::max(std::min((float)round(radii), 5.0f), 1.0f);
        bool invalid = false;

        float3 e1 = getTubeDirection(T, pos, size);

        for(int a = -maxD; a <= maxD; a++) {
        for(int b = -maxD; b <= maxD; b++) {
        for(int c = -maxD; c <= maxD; c++) {
            if(a == 0 && b == 0 && c == 0)
                continue;
            float3 r(a,b,c);
            float length = r.length();
            int3 n(pos.x + r.x,pos.y+r.y,pos.z+r.z);
            if(!inBounds(n,size))
                continue;
            float dp = e1.dot(r);
            float3 r_projected(r.x-e1.x*dp,r.y-e1.y*dp,r.z-e1.z*dp);
            float3 rn = r.normalize();
            float3 r_projected_n = r_projected.normalize();
            float theta = acos(rn.dot(r_projected_n));
            if((theta < thetaLimit && length < maxD)) {
                if(SQR_MAG(n) < SQR_MAG(pos)) {
                    invalid = true;
                    break;
                }

            }
        }}}
        if(!invalid) {
#pragma omp critical
            filteredPoints.insert(POS(pos));
        }
    }
    candidatePoints.clear();
    std::cout << "filtered points: " << filteredPoints.size() << std::endl;

    std::vector<int3> centerpoints;
    for(int z = 0; z < size.z/cubeSize; z++) {
    for(int y = 0; y < size.y/cubeSize; y++) {
    for(int x = 0; x < size.x/cubeSize; x++) {
        int3 bestPos;
        float bestTDF = 0.0f;
        int3 readPos(
            x*cubeSize,
            y*cubeSize,
            z*cubeSize
        );
        bool found = false;
        for(int a = 0; a < cubeSize; a++) {
        for(int b = 0; b < cubeSize; b++) {
        for(int c = 0; c < cubeSize; c++) {
            int3 pos = readPos + int3(a,b,c);
            if(filteredPoints.find(POS(pos)) != filteredPoints.end()) {
                float tdf = T.TDF[POS(pos)];
                if(tdf > bestTDF) {
                    found = true;
                    bestTDF = tdf;
                    bestPos = pos;
                }
            }
        }}}
        if(found) {
#pragma omp critical
            centerpoints.push_back(bestPos);
        }
    }}}

    int points = centerpoints.size();
    std::cout << "filtered points: " <<points<< std::endl;
    std::vector<SIPL::int2> edges;

    // Do linking
    for(int i = 0; i < points;i++) {
        int3 xa = centerpoints[i];
        SIPL::int2 bestPair;
        float shortestDistance = maxDistance*2;
        bool validPairFound = false;

        for(int j = 0; j < points;j++) {
            if(i == j)
                continue;
            int3 xb = centerpoints[j];

            int db = round(xa.distance(xb));
            if(db >= shortestDistance)
                continue;
            for(int k = 0; k < j;k++) {
                if(k == i)
                    continue;
                int3 xc = centerpoints[k];

                int dc = round(xa.distance(xc));

                if(db+dc < shortestDistance) {
                    // Check angle
                    int3 ab = (xb-xa);
                    int3 ac = (xc-xa);
                    float angle = acos(ab.normalize().dot(ac.normalize()));
                    //printf("angle: %f\n", angle);
                    if(angle < 2.0f) // 120 degrees
                    //if(angle < 1.57f) // 90 degrees
                        continue;

                    // Check avg TDF for a-b
                    float avgTDF = 0.0f;
                    for(int l = 0; l <= db; l++) {
                        float alpha = (float)l/db;
                        int3 p((int)round(xa.x+ab.x*alpha),(int)round(xa.y+ab.y*alpha),(int)round(xa.z+ab.z*alpha));
                        float t = T.TDF[POS(p)];
                        avgTDF += t;
                    }
                    avgTDF /= db+1;
                    if(avgTDF < minAvgTDF)
                        continue;

                    avgTDF = 0.0f;

                    // Check avg TDF for a-c
                    for(int l = 0; l <= dc; l++) {
                        float alpha = (float)l/dc;
                        int3 p((int)round(xa.x+ac.x*alpha),(int)round(xa.y+ac.y*alpha),(int)round(xa.z+ac.z*alpha));
                        float t = T.TDF[POS(p)];
                        avgTDF += t;
                    }
                    avgTDF /= dc+1;

                    if(avgTDF < minAvgTDF)
                        continue;

                    validPairFound = true;
                    bestPair.x = j;
                    bestPair.y = k;
                    shortestDistance = db+dc;
                }
            } // k
        }// j

        if(validPairFound) {
            // Store edges
            SIPL::int2 edge(i, bestPair.x);
            SIPL::int2 edge2(i, bestPair.y);
            edges.push_back(edge);
            edges.push_back(edge2);
        }
    } // i
    std::cout << "nr of edges: " << edges.size() << std::endl;

    // Do graph component labeling

    // Select wanted parts of centerline

    // Create VTK file
    // Create centerline image

    /*
    std::vector<SIPL::int2> edges;
    int maxEdgeDistance = getParam(parameters, "max-edge-distance");
    for(int i = 0; i < sum2; i++) {
        if(SArray[CArray[edgesArray[i*2]]] >= minTreeLength && SArray[CArray[edgesArray[i*2+1]]] >= minTreeLength ) {
            // Check length of edge
            int3 A = vertices[indexes[edgesArray[i*2]]];
            int3 B = vertices[indexes[edgesArray[i*2+1]]];
            float distance = A.distance(B);
            if(getParamStr(parameters, "centerline-vtk-file") != "off" &&
                    distance > maxEdgeDistance) {
                float3 direction(B.x-A.x,B.y-A.y,B.z-A.z);
                float3 Af(A.x,A.y,A.z);
                int previous = indexes[edgesArray[i*2]];
                for(int j = maxEdgeDistance; j < distance; j += maxEdgeDistance) {
                    float3 newPos = Af + ((float)j/distance)*direction;
                    int3 newVertex(round(newPos.x), round(newPos.y), round(newPos.z));
                    // Create new vertex
                    vertices.push_back(newVertex);
                    // Add new edge
                    SIPL::int2 edge(previous, counter);
                    edges.push_back(edge);
                    previous = counter;
                    counter++;
                }
                // Connect previous vertex to B
                SIPL::int2 edge(previous, indexes[edgesArray[i*2+1]]);
                edges.push_back(edge);
            } else {
                SIPL::int2 v(indexes[edgesArray[i*2]],indexes[edgesArray[i*2+1]]);
                edges.push_back(v);
            }
        }
    }
    */
    std::vector<int3> vertices = centerpoints;

    // Remove loops from graph
    if(getParamBool(parameters, "loop-removal"))
        removeLoops(vertices, edges, size);

    ocl.queue.finish();
    char * centerlinesData = createCenterlineVoxels(vertices, edges, T.radius, size);
    Image3D centerlines= Image3D(
        ocl.context,
        CL_MEM_READ_WRITE,
        ImageFormat(CL_R, CL_SIGNED_INT8),
        size.x, size.y, size.z
    );

    ocl.queue.enqueueWriteImage(
            centerlines,
            CL_FALSE,
            offset,
            region,
            0, 0,
            centerlinesData
    );
    ocl.queue.enqueueWriteImage(
            radius,
            CL_FALSE,
            offset,
            region,
            0, 0,
            T.radius
    );

    if(getParamStr(parameters, "centerline-vtk-file") != "off") {
        writeToVtkFile(parameters, vertices, edges);
    }

    ocl.queue.finish();

    delete[] T.TDF;
    delete[] T.Fx;
    delete[] T.Fy;
    delete[] T.Fz;
    delete[] T.radius;
    delete[] centerlinesData;

    return centerlines;
}

Image3D runNewCenterlineAlg(OpenCL &ocl, SIPL::int3 size, paramList &parameters, Image3D &vectorField, Image3D &TDF, Image3D &radius) {
    if(ocl.platform.getInfo<CL_PLATFORM_VENDOR>().substr(0,5) == "Apple") {
        std::cout << "Apple platform detected. Running centerline extraction without OpenCL." << std::endl;
        return runNewCenterlineAlgWithoutOpenCL(ocl,size,parameters,vectorField,TDF,radius);
    }
    const int totalSize = size.x*size.y*size.z;
	const bool no3Dwrite = !getParamBool(parameters, "3d_write");
    const int cubeSize = getParam(parameters, "cube-size");
    const int minTreeLength = getParam(parameters, "min-tree-length");
    const float Thigh = getParam(parameters, "tdf-high");
    const float Tmean = getParam(parameters, "min-mean-tdf");
    const float maxDistance = getParam(parameters, "max-distance");

    cl::size_t<3> offset;
    offset[0] = 0;
    offset[1] = 0;
    offset[2] = 0;
    cl::size_t<3> region;
    region[0] = size.x;
    region[1] = size.y;
    region[2] = size.z;

    Kernel candidatesKernel(ocl.program, "findCandidateCenterpoints");
    Kernel candidates2Kernel(ocl.program, "findCandidateCenterpoints2");
    Kernel ddKernel(ocl.program, "dd");
    Kernel initCharBuffer(ocl.program, "initCharBuffer");

    cl::Event startEvent, endEvent;
    cl_ulong start, end;
    if(getParamBool(parameters, "timing")) {
        ocl.queue.enqueueMarker(&startEvent);
    }
    Image3D * centerpointsImage2 = new Image3D(
            ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_R, CL_SIGNED_INT8),
            size.x, size.y, size.z
    );
    GC->addMemObject(centerpointsImage2);
    Buffer vertices;
    int sum = 0;

    if(no3Dwrite) {
        Buffer * centerpoints = new Buffer(
                ocl.context,
                CL_MEM_READ_WRITE,
                sizeof(char)*totalSize
        );
        GC->addMemObject(centerpoints);

        candidatesKernel.setArg(0, TDF);
        candidatesKernel.setArg(1, *centerpoints);
        candidatesKernel.setArg(2, Thigh);
        ocl.queue.enqueueNDRangeKernel(
                candidatesKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NullRange
        );

        HistogramPyramid3DBuffer hp3(ocl);
        hp3.create(*centerpoints, size.x, size.y, size.z);

        candidates2Kernel.setArg(0, TDF);
        candidates2Kernel.setArg(1, radius);
        candidates2Kernel.setArg(2, vectorField);
        Buffer * centerpoints2 = new Buffer(
                ocl.context,
                CL_MEM_READ_WRITE,
                sizeof(char)*totalSize
        );
        GC->addMemObject(centerpoints2);
        initCharBuffer.setArg(0, *centerpoints2);
        ocl.queue.enqueueNDRangeKernel(
                initCharBuffer,
                NullRange,
                NDRange(totalSize),
                NullRange
        );

        candidates2Kernel.setArg(3, *centerpoints2);
        std::cout << "candidates: " << hp3.getSum() << std::endl;
        if(hp3.getSum() <= 0 || hp3.getSum() > 0.5*totalSize) {
        	throw SIPL::SIPLException("The number of candidate voxels is too low or too high. Something went wrong... Wrong parameters? Out of memory?", __LINE__, __FILE__);
        }
        hp3.traverse(candidates2Kernel, 4);
        ocl.queue.finish();
        hp3.deleteHPlevels();
        GC->deleteMemObject(centerpoints);
        ocl.queue.enqueueCopyBufferToImage(
            *centerpoints2,
            *centerpointsImage2,
            0,
            offset,
            region
        );
        ocl.queue.finish();
        GC->deleteMemObject(centerpoints2);

		if(getParamBool(parameters, "centerpoints-only")) {
			return *centerpointsImage2;
		}
        ddKernel.setArg(0, TDF);
        ddKernel.setArg(1, *centerpointsImage2);
        ddKernel.setArg(3, cubeSize);
        Buffer * centerpoints3 = new Buffer(
                ocl.context,
                CL_MEM_READ_WRITE,
                sizeof(char)*totalSize
        );
        GC->addMemObject(centerpoints3);
        initCharBuffer.setArg(0, *centerpoints3);
        ocl.queue.enqueueNDRangeKernel(
                initCharBuffer,
                NullRange,
                NDRange(totalSize),
                NullRange
        );
        ddKernel.setArg(2, *centerpoints3);
        ocl.queue.enqueueNDRangeKernel(
                ddKernel,
                NullRange,
                NDRange(ceil((float)size.x/cubeSize),ceil((float)size.y/cubeSize),ceil((float)size.z/cubeSize)),
                NullRange
        );
        ocl.queue.finish();
        GC->deleteMemObject(centerpointsImage2);

        // Construct HP of centerpointsImage
        HistogramPyramid3DBuffer hp(ocl);
        hp.create(*centerpoints3, size.x, size.y, size.z);
        sum = hp.getSum();
        std::cout << "number of vertices detected " << sum << std::endl;

        // Run createPositions kernel
        vertices = hp.createPositionBuffer();
        ocl.queue.finish();
        hp.deleteHPlevels();
        GC->deleteMemObject(centerpoints3);
    } else {
        Kernel init3DImage(ocl.program, "init3DImage");
        init3DImage.setArg(0, *centerpointsImage2);
        ocl.queue.enqueueNDRangeKernel(
            init3DImage,
            NullRange,
            NDRange(size.x,size.y,size.z),
            NullRange
        );

        Image3D * centerpointsImage = new Image3D(
                ocl.context,
                CL_MEM_READ_WRITE,
                ImageFormat(CL_R, CL_SIGNED_INT8),
                size.x, size.y, size.z
        );
        GC->addMemObject(centerpointsImage);

        candidatesKernel.setArg(0, TDF);
        candidatesKernel.setArg(1, *centerpointsImage);
        candidatesKernel.setArg(2, Thigh);
        ocl.queue.enqueueNDRangeKernel(
                candidatesKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NDRange(4,4,4)
        );


        candidates2Kernel.setArg(0, TDF);
        candidates2Kernel.setArg(1, radius);
        candidates2Kernel.setArg(2, vectorField);

        HistogramPyramid3D hp3(ocl);
        hp3.create(*centerpointsImage, size.x, size.y, size.z);
        std::cout << "candidates: " << hp3.getSum() << std::endl;
		if(hp3.getSum() <= 0 || hp3.getSum() > 0.5*totalSize) {
        	throw SIPL::SIPLException("The number of candidate voxels is too or too high. Something went wrong... Wrong parameters? Out of memory?", __LINE__, __FILE__);
        }

        candidates2Kernel.setArg(3, *centerpointsImage2);
        hp3.traverse(candidates2Kernel, 4);
        ocl.queue.finish();
        hp3.deleteHPlevels();
        GC->deleteMemObject(centerpointsImage);

        Image3D * centerpointsImage3 = new Image3D(
                ocl.context,
                CL_MEM_READ_WRITE,
                ImageFormat(CL_R, CL_SIGNED_INT8),
                size.x, size.y, size.z
        );
        GC->addMemObject(centerpointsImage3);
        init3DImage.setArg(0, *centerpointsImage3);
        ocl.queue.enqueueNDRangeKernel(
            init3DImage,
            NullRange,
            NDRange(size.x,size.y,size.z),
            NullRange
        );

		if(getParamBool(parameters, "centerpoints-only")) {
			return *centerpointsImage2;
		}
        ddKernel.setArg(0, TDF);
        ddKernel.setArg(1, *centerpointsImage2);
        ddKernel.setArg(3, cubeSize);
        ddKernel.setArg(2, *centerpointsImage3);
        ocl.queue.enqueueNDRangeKernel(
                ddKernel,
                NullRange,
                NDRange(ceil((float)size.x/cubeSize),ceil((float)size.y/cubeSize),ceil((float)size.z/cubeSize)),
                NullRange
        );
        ocl.queue.finish();
        GC->deleteMemObject(centerpointsImage2);

        // Construct HP of centerpointsImage
        HistogramPyramid3D hp(ocl);
        hp.create(*centerpointsImage3, size.x, size.y, size.z);
        sum = hp.getSum();
        std::cout << "number of vertices detected " << sum << std::endl;

        // Run createPositions kernel
        vertices = hp.createPositionBuffer();
        ocl.queue.finish();
        hp.deleteHPlevels();
        GC->deleteMemObject(centerpointsImage3);
    }
    if(sum < 8) {
    	throw SIPL::SIPLException("Too few centerpoints detected. Revise parameters.", __LINE__, __FILE__);
    } else if(sum >= 16384) {
    	throw SIPL::SIPLException("Too many centerpoints detected. More cropping of dataset is probably needed.", __LINE__, __FILE__);
    }

if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&endEvent);
    ocl.queue.finish();
    startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    endEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &end);
    std::cout << "RUNTIME centerpoint extraction: " << (end-start)*1.0e-6 << " ms" << std::endl;
}

if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&startEvent);
}
    // Run linking kernel
    Image2D edgeTuples = Image2D(
            ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_R, CL_UNSIGNED_INT8),
            sum, sum
    );
    Kernel init2DImage(ocl.program, "init2DImage");
    init2DImage.setArg(0, edgeTuples);
    ocl.queue.enqueueNDRangeKernel(
        init2DImage,
        NullRange,
        NDRange(sum, sum),
        NullRange
    );
    int globalSize = sum;
    while(globalSize % 64 != 0) globalSize++;

    // Create lengths image
    Image2D * lengths = new Image2D(
            ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_R, CL_FLOAT),
            sum, sum
    );
    GC->addMemObject(lengths);

    // Run linkLengths kernel
    Kernel linkLengths(ocl.program, "linkLengths");
    linkLengths.setArg(0, vertices);
    linkLengths.setArg(1, *lengths);
    ocl.queue.enqueueNDRangeKernel(
            linkLengths,
            NullRange,
            NDRange(sum, sum),
            NullRange
    );

    // Create and init compacted_lengths image
    float * cl = new float[sum*sum*2]();
    Image2D * compacted_lengths = new Image2D(
            ocl.context,
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            ImageFormat(CL_RG, CL_FLOAT),
            sum, sum,
            0,
            cl
    );
    GC->addMemObject(compacted_lengths);
    delete[] cl;

    // Create and initialize incs buffer
    Buffer incs = Buffer(
            ocl.context,
            CL_MEM_READ_WRITE,
            sizeof(int)*sum
    );

    Kernel initIncsBuffer(ocl.program, "initIntBuffer");
    initIncsBuffer.setArg(0, incs);
    ocl.queue.enqueueNDRangeKernel(
        initIncsBuffer,
        NullRange,
        NDRange(sum),
        NullRange
    );

    // Run compact kernel
    Kernel compactLengths(ocl.program, "compact");
    compactLengths.setArg(0, *lengths);
    compactLengths.setArg(1, incs);
    compactLengths.setArg(2, *compacted_lengths);
    compactLengths.setArg(3, maxDistance);
    ocl.queue.enqueueNDRangeKernel(
            compactLengths,
            NullRange,
            NDRange(sum, sum),
            NullRange
    );
    ocl.queue.finish();
    GC->deleteMemObject(lengths);

    Kernel linkingKernel(ocl.program, "linkCenterpoints");
    linkingKernel.setArg(0, TDF);
    linkingKernel.setArg(1, vertices);
    linkingKernel.setArg(2, edgeTuples);
    linkingKernel.setArg(3, *compacted_lengths);
    linkingKernel.setArg(4, sum);
    linkingKernel.setArg(5, Tmean);
    linkingKernel.setArg(6, maxDistance);
    ocl.queue.enqueueNDRangeKernel(
            linkingKernel,
            NullRange,
            NDRange(globalSize),
            NDRange(64)
    );
    ocl.queue.finish();
    GC->deleteMemObject(compacted_lengths);
if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&endEvent);
    ocl.queue.finish();
    startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    endEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &end);
    std::cout << "RUNTIME linking: " << (end-start)*1.0e-6 << " ms" << std::endl;
}

if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&startEvent);
}

	// Remove duplicate edges
	Image2D edgeTuples2 = Image2D(
			ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_R, CL_UNSIGNED_INT8),
            sum, sum
    );
	Kernel removeDuplicatesKernel(ocl.program, "removeDuplicateEdges");
	removeDuplicatesKernel.setArg(0, edgeTuples);
	removeDuplicatesKernel.setArg(1, edgeTuples2);
	ocl.queue.enqueueNDRangeKernel(
			removeDuplicatesKernel,
			NullRange,
			NDRange(sum,sum),
			NullRange
	);
	edgeTuples = edgeTuples2;

    // Run HP on edgeTuples
    HistogramPyramid2D hp2(ocl);
    hp2.create(edgeTuples, sum, sum);

	std::cout << "number of edges detected " << hp2.getSum() << std::endl;
    if(hp2.getSum() == 0) {
        throw SIPL::SIPLException("No edges were found", __LINE__, __FILE__);
    } else if(hp2.getSum() > 10000000) {
    	throw SIPL::SIPLException("More than 10 million edges found. Must be wrong!", __LINE__, __FILE__);
    } else if(hp2.getSum() < 0){
    	throw SIPL::SIPLException("A negative number of edges was found!", __LINE__, __FILE__);
    }

    // Run create positions kernel on edges
    Buffer edges = hp2.createPositionBuffer();
    ocl.queue.finish();
    hp2.deleteHPlevels();
if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&endEvent);
    ocl.queue.finish();
    startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    endEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &end);
    std::cout << "RUNTIME HP creation and traversal: " << (end-start)*1.0e-6 << " ms" << std::endl;
}

if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&startEvent);
}

    // Do graph component labeling
    Buffer C = Buffer(
            ocl.context,
            CL_MEM_READ_WRITE,
            sizeof(int)*sum
    );
    Kernel initCBuffer(ocl.program, "initIntBufferID");
    initCBuffer.setArg(0, C);
    initCBuffer.setArg(1, sum);
    ocl.queue.enqueueNDRangeKernel(
        initCBuffer,
        NullRange,
        NDRange(globalSize),
        NDRange(64)
    );


    Buffer m = Buffer(
            ocl.context,
            CL_MEM_WRITE_ONLY,
            sizeof(int)
    );

    Kernel labelingKernel(ocl.program, "graphComponentLabeling");
    labelingKernel.setArg(0, edges);
    labelingKernel.setArg(1, C);
    labelingKernel.setArg(2, m);
    int M;
    int sum2 = hp2.getSum();
    labelingKernel.setArg(3, sum2);
    globalSize = sum2;
    while(globalSize % 64 != 0) globalSize++;
    int i = 0;
    int minIterations = 100;
    do {
        // write 0 to m
        if(i > minIterations) {
            M = 0;
            ocl.queue.enqueueWriteBuffer(m, CL_FALSE, 0, sizeof(int), &M);
        } else {
            M = 1;
        }

        ocl.queue.enqueueNDRangeKernel(
                labelingKernel,
                NullRange,
                NDRange(globalSize),
                NDRange(64)
        );

        // read m from device
        if(i > minIterations)
            ocl.queue.enqueueReadBuffer(m, CL_TRUE, 0, sizeof(int), &M);
        ++i;
    } while(M == 1);
    std::cout << "did graph component labeling in " << i << " iterations " << std::endl;
if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&endEvent);
    ocl.queue.finish();
    startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    endEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &end);
    std::cout << "RUNTIME graph component labeling: " << (end-start)*1.0e-6 << " ms" << std::endl;
}


if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&startEvent);
}
    // Remove small trees
    Buffer S = Buffer(
            ocl.context,
            CL_MEM_READ_WRITE,
            sizeof(int)*sum
    );
    Kernel initIntBuffer(ocl.program, "initIntBuffer");
    initIntBuffer.setArg(0, S);
    ocl.queue.enqueueNDRangeKernel(
        initIntBuffer,
        NullRange,
        NDRange(sum),
        NullRange
    );
    Kernel calculateTreeLengthKernel(ocl.program, "calculateTreeLength");
    calculateTreeLengthKernel.setArg(0, C);
    calculateTreeLengthKernel.setArg(1, S);

    ocl.queue.enqueueNDRangeKernel(
            calculateTreeLengthKernel,
            NullRange,
            NDRange(sum),
            NullRange
    );
    Image3D centerlines= Image3D(
            ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_R, CL_SIGNED_INT8),
            size.x, size.y, size.z
        );

    if(getParamStr(parameters, "centerline-vtk-file") != "off" ||
    		getParamBool(parameters, "loop-removal")) {
    	// Do rasterization of centerline on CPU
    	// Transfer edges (size: sum2) and vertices (size: sum) buffers to host
    	int * verticesArray = new int[sum*3];
    	int * edgesArray = new int[sum2*2];
    	int * CArray = new int[sum];
    	int * SArray = new int[sum];

    	ocl.queue.enqueueReadBuffer(vertices, CL_FALSE, 0, sum*3*sizeof(int), verticesArray);
    	ocl.queue.enqueueReadBuffer(edges, CL_FALSE, 0, sum2*2*sizeof(int), edgesArray);
    	ocl.queue.enqueueReadBuffer(C, CL_FALSE, 0, sum*sizeof(int), CArray);
    	ocl.queue.enqueueReadBuffer(S, CL_FALSE, 0, sum*sizeof(int), SArray);

    	ocl.queue.finish();
    	float * radiusB = new float[totalSize];
    	ocl.queue.enqueueReadImage(radius, CL_FALSE, offset, region, 0, 0, radiusB);
    	std::vector<int3> vertices;
    	int counter = 0;
    	int * indexes = new int[sum];
    	for(int i = 0; i < sum; i++) {
    		if(SArray[CArray[i]] >= minTreeLength) {
				int3 v(verticesArray[i*3],verticesArray[i*3+1],verticesArray[i*3+2]);
				vertices.push_back(v);
				indexes[i] = counter;
				counter++;
    		}
    	}
    	std::vector<SIPL::int2> edges;
		int maxEdgeDistance = getParam(parameters, "max-edge-distance");
    	for(int i = 0; i < sum2; i++) {
    		if(SArray[CArray[edgesArray[i*2]]] >= minTreeLength && SArray[CArray[edgesArray[i*2+1]]] >= minTreeLength ) {
    			// Check length of edge
    			int3 A = vertices[indexes[edgesArray[i*2]]];
    			int3 B = vertices[indexes[edgesArray[i*2+1]]];
    			float distance = A.distance(B);
    			if(getParamStr(parameters, "centerline-vtk-file") != "off" &&
    					distance > maxEdgeDistance) {
					float3 direction(B.x-A.x,B.y-A.y,B.z-A.z);
					float3 Af(A.x,A.y,A.z);
					int previous = indexes[edgesArray[i*2]];
    				for(int j = maxEdgeDistance; j < distance; j += maxEdgeDistance) {
    					float3 newPos = Af + ((float)j/distance)*direction;
    					int3 newVertex(round(newPos.x), round(newPos.y), round(newPos.z));
    					// Create new vertex
    					vertices.push_back(newVertex);
    					// Add new edge
    					SIPL::int2 edge(previous, counter);
    					edges.push_back(edge);
    					previous = counter;
    					counter++;
    				}
    				// Connect previous vertex to B
    				SIPL::int2 edge(previous, indexes[edgesArray[i*2+1]]);
    				edges.push_back(edge);
    			} else {
					SIPL::int2 v(indexes[edgesArray[i*2]],indexes[edgesArray[i*2+1]]);
					edges.push_back(v);
    			}
    		}
    	}

    	// Remove loops from graph
    	removeLoops(vertices, edges, size);

    	ocl.queue.finish();
    	char * centerlinesData = createCenterlineVoxels(vertices, edges, radiusB, size);
    	ocl.queue.enqueueWriteImage(
    			centerlines,
    			CL_FALSE,
    			offset,
    			region,
    			0, 0,
    			centerlinesData
		);
		ocl.queue.enqueueWriteImage(
    			radius,
    			CL_FALSE,
    			offset,
    			region,
    			0, 0,
    			radiusB
		);

		if(getParamStr(parameters, "centerline-vtk-file") != "off")
			writeToVtkFile(parameters, vertices, edges);

    	delete[] verticesArray;
    	delete[] edgesArray;
    	delete[] CArray;
    	delete[] SArray;
    	delete[] indexes;
    } else {
    	// Do rasterization of centerline on GPU
    	Kernel RSTKernel(ocl.program, "removeSmallTrees");
		RSTKernel.setArg(0, edges);
		RSTKernel.setArg(1, vertices);
		RSTKernel.setArg(2, C);
		RSTKernel.setArg(3, S);
		RSTKernel.setArg(4, minTreeLength);
		if(no3Dwrite) {
			Buffer centerlinesBuffer = Buffer(
					ocl.context,
					CL_MEM_WRITE_ONLY,
					sizeof(char)*totalSize
			);

			initCharBuffer.setArg(0, centerlinesBuffer);
			ocl.queue.enqueueNDRangeKernel(
					initCharBuffer,
					NullRange,
					NDRange(totalSize),
					NullRange
			);

			RSTKernel.setArg(5, centerlinesBuffer);
			RSTKernel.setArg(6, size.x);
			RSTKernel.setArg(7, size.y);

			ocl.queue.enqueueNDRangeKernel(
					RSTKernel,
					NullRange,
					NDRange(sum2),
					NullRange
			);

			ocl.queue.enqueueCopyBufferToImage(
					centerlinesBuffer,
					centerlines,
					0,
					offset,
					region
			);

		} else {

			Kernel init3DImage(ocl.program, "init3DImage");
			init3DImage.setArg(0, centerlines);
			ocl.queue.enqueueNDRangeKernel(
				init3DImage,
				NullRange,
				NDRange(size.x, size.y, size.z),
				NullRange
			);

			RSTKernel.setArg(5, centerlines);

			ocl.queue.enqueueNDRangeKernel(
					RSTKernel,
					NullRange,
					NDRange(sum2),
					NullRange
			);
		}
    }

if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&endEvent);
    ocl.queue.finish();
    startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    endEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &end);
    std::cout << "RUNTIME of removing small trees: " << (end-start)*1.0e-6 << " ms" << std::endl;
}
    return centerlines;
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

class CrossSection {
public:
	int3 pos;
	float TDF;
	std::vector<CrossSection *> neighbors;
	int label;
	int index;
	float3 direction;
};

class CrossSectionCompare {
private:
	float * dist;
public:
	CrossSectionCompare(float * dist) { this->dist = dist; };
	bool operator() (const CrossSection * a, const CrossSection * b) {
		return dist[a->index] > dist[b->index];
	};
};

std::vector<CrossSection *> createGraph(TubeSegmentation &T, SIPL::int3 size) {
	// Create vector
	std::vector<CrossSection *> sections;
	float threshold = 0.5f;

	// Go through TS.TDF and add all with TDF above threshold
	int counter = 0;
	float thetaLimit = 0.5;
	for(int z = 1; z < size.z-1; z++) {
	for(int y = 1; y < size.y-1; y++) {
	for(int x = 1; x < size.x-1; x++) {
		int3 pos(x,y,z);
		float tdf = T.TDF[POS(pos)];
		if(tdf > threshold) {
			int maxD = std::min(std::max((int)round(T.radius[POS(pos)]), 1), 5);
		    //std::cout << SQR_MAG(pos) << " " << SQR_MAG_SMALL(pos) << std::endl;
			//std::cout << "radius" << TS.radius[POS(pos)] << std::endl;
			//std::cout << "maxD "<< maxD <<std::endl;
			float3 e1 = getTubeDirection(T, pos, size);
			bool invalid = false;
		    for(int a = -maxD; a <= maxD; a++) {
		    for(int b = -maxD; b <= maxD; b++) {
		    for(int c = -maxD; c <= maxD; c++) {
		        if(a == 0 && b == 0 && c == 0)
		            continue;
		        const int3 n = pos + int3(a,b,c);
		        if(!inBounds(n, size))
		        	continue;
		        float3 r(a,b,c);
		        const float dp = e1.dot(r);
		        float3 r_projected = float3(r.x-e1.x*dp,r.y-e1.y*dp,r.z-e1.z*dp);
		        float3 rn = r.normalize();
		        float3 rpn = r_projected.normalize();
		        float theta = acos((double)rn.dot(rpn));
		        //std::cout << "theta: " << theta << std::endl;

		        if((theta < thetaLimit && r.length() < maxD-0.5f)) {
		        	//std::cout << SQR_MAG(n) << std::endl;
		        	if(T.radius[POS(pos)]<= 3) {
		            //if(TS.TDF[POS(n)] > TS.TDF[POS(pos)]) {
		            if(SQR_MAG_SMALL(n) < SQR_MAG_SMALL(pos)) {
		                invalid = true;
		                break;
		            }
		            } else {
		            if(SQR_MAG(n) < SQR_MAG(pos)) {
		            //if(TS.TDF[POS(n)] > TS.TDF[POS(pos)]) {
		                invalid = true;
		                break;
		            }
		            }
		        }
		    }}}
		    if(!invalid) {
				CrossSection * cs = new CrossSection;
				cs->pos = pos;
				cs->TDF = tdf;
				cs->label = -1;//counter;
				cs->direction = e1;
				counter++;
				sections.push_back(cs);
		    }
		}
	}}}


	// For each cross section c_i
	for(int i = 0; i < sections.size(); i++) {
		CrossSection * c_i = sections[i];
		// For each cross section c_j
		for(int j = 0; j < i; j++) {
			CrossSection * c_j = sections[j];
			// If all criterias are ok: Add c_j as neighbor to c_i
			if(c_i->pos.distance(c_j->pos) < 4 && !(c_i->pos == c_j->pos)) {
				float3 e1_i = c_i->direction;
				float3 e1_j = c_j->direction;
				int3 cint = c_i->pos - c_j->pos;
				float3 c = cint.normalize();

				if(acos((double)fabs(e1_i.dot(e1_j))) > 1.05) // 60 degrees
					continue;

				if(acos((double)fabs(e1_i.dot(c))) > 1.05)
					continue;

				if(acos((double)fabs(e1_j.dot(c))) > 1.05)
					continue;

				int distance = ceil(c_i->pos.distance(c_j->pos));
				float3 direction(c_j->pos.x-c_i->pos.x,c_j->pos.y-c_i->pos.y,c_j->pos.z-c_i->pos.z);
				bool invalid = false;
				for(int i = 0; i < distance; i++) {
					float frac = (float)i/distance;
					float3 n = c_i->pos + frac*direction;
					int3 in(round(n.x),round(n.y),round(n.z));
					//float3 e1 = getTubeDirection(TS, in, size);
					//cost += (1-fabs(a->direction.dot(e1)))+(1-fabs(b->direction.dot(e1)));
					/*
					if(T.intensity[POS(in)] > 0.5f) {
						invalid = true;
						break;
					}
					*/
				}
				//if(invalid)
				//	continue;


				c_i->neighbors.push_back(c_j);
				c_j->neighbors.push_back(c_i);
				//sectionPairs.push_back(c_i);
			}
			// If no pair is found, dont add it
		}
	}

	std::vector<CrossSection *> sectionPairs;
	for(int i = 0; i < sections.size(); i++) {
		CrossSection * c_i = sections[i];
		if(c_i->neighbors.size()>0) {
			sectionPairs.push_back(c_i);
		}
	}

	return sectionPairs;
}

class Connection;

class Segment {
public:
	std::vector<CrossSection *> sections;
	std::vector<Connection *> connections;
	float benefit;
	float cost;
	int index;
};

class Connection {
public:
	Segment * source;
	Segment * target;
	CrossSection * source_section;
	CrossSection * target_section;
	float cost;
};

bool segmentCompare(Segment * a, Segment * b) {
	return a->benefit > b->benefit;
}

bool segmentInSegmentation(Segment * s, unordered_set<int> &segmentation, int3 size) {
	bool in = false;
	for(int i = 0; i < s->sections.size(); i++) {
		CrossSection * c = s->sections[i];
		if(segmentation.find(POS(c->pos)) != segmentation.end()) {
			in = true;
			break;
		}
	}
	return in;
}

float calculateBenefit(CrossSection * a, CrossSection * b, TubeSegmentation &TS, int3 size) {
	float benefit = 0.0f;
	int distance = ceil(a->pos.distance(b->pos));
	float3 direction(b->pos.x-a->pos.x,b->pos.y-a->pos.y,b->pos.z-a->pos.z);
	for(int i = 0; i < distance; i++) {
		float frac = (float)i/distance;
		float3 n = a->pos + frac*direction;
		int3 in(round(n.x),round(n.y),round(n.z));
		benefit += (TS.TDF[POS(in)]);
	}

	return benefit;
}

void inverseGradientRegionGrowing(Segment * s, TubeSegmentation &TS, unordered_set<int> &segmentation, int3 size) {
    std::vector<int3> centerpoints;
	for(int c = 0; c < s->sections.size()-1; c++) {
		CrossSection * a = s->sections[c];
		CrossSection * b = s->sections[c+1];
		int distance = ceil(a->pos.distance(b->pos));
		float3 direction(b->pos.x-a->pos.x,b->pos.y-a->pos.y,b->pos.z-a->pos.z);
		for(int i = 0; i < distance; i++) {
			float frac = (float)i/distance;
			float3 n = a->pos + frac*direction;
			int3 in(round(n.x),round(n.y),round(n.z));
			centerpoints.push_back(in);
		}
		segmentation.insert(POS(a->pos));//test
		segmentation.insert(POS(b->pos));//test
	}

	// Dilate the centerline
	std::vector<int3> dilatedCenterline;
	for(int i = 0; i < centerpoints.size(); i++) {
		int3 pos = centerpoints[i];
		for(int a = -1; a < 2; a++) {
		for(int b = -1; b < 2; b++) {
		for(int c = -1; c < 2; c++) {
			int3 n = pos + int3(a,b,c);
			if(inBounds(n, size) && segmentation.find(POS(n)) == segmentation.end()) {
				segmentation.insert(POS(n));
				dilatedCenterline.push_back(n);
			}
		}}}
	}
	/*

	std::queue<int3> queue;
	for(int3 pos : dilatedCenterline) {
		for(int a = -1; a < 2; a++) {
		for(int b = -1; b < 2; b++) {
		for(int c = -1; c < 2; c++) {
			int3 n = pos + int3(a,b,c);
			if(inBounds(n, size) && segmentation.find(POS(n)) == segmentation.end()) {
				queue.push(n);
			}
		}}}
	}

	while(!queue.empty()) {
		int3 X = queue.front();
		float FNXw = SQR_MAG(X);
		queue.pop();
		for(int a = -1; a < 2; a++) {
		for(int b = -1; b < 2; b++) {
		for(int c = -1; c < 2; c++) {
			if(a == 0 && b == 0 && c == 0)
				continue;

			int3 Y = X + int3(a,b,c);
			if(inBounds(Y, size) && segmentation.find(POS(Y)) == segmentation.end()) {

				float3 FNY;
				FNY.x = TS.Fx[POS(Y)];
				FNY.y = TS.Fy[POS(Y)];
				FNY.z = TS.Fz[POS(Y)];
				float FNYw = FNY.length();
				FNY = FNY.normalize();
				if(FNYw > FNXw || FNXw < 0.1f) {

					int3 Z;
					float maxDotProduct = -2.0f;
					for(int a2 = -1; a2 < 2; a2++) {
					for(int b2 = -1; b2 < 2; b2++) {
					for(int c2 = -1; c2 < 2; c2++) {
						if(a2 == 0 && b2 == 0 && c2 == 0)
							continue;
						int3 Zc;
						Zc.x = Y.x+a2;
						Zc.y = Y.y+b2;
						Zc.z = Y.z+c2;
						float3 YZ;
						YZ.x = Zc.x-Y.x;
						YZ.y = Zc.y-Y.y;
						YZ.z = Zc.z-Y.z;
						YZ = YZ.normalize();
						if(FNY.dot(YZ) > maxDotProduct) {
							maxDotProduct = FNY.dot(YZ);
							Z = Zc;
						}
					}}}

					if(Z.x == X.x && Z.y == X.y && Z.z == X.z) {
						segmentation.insert(POS(X));
						queue.push(Y);
					}
				}
			}
		}}}
	}
	*/

}

std::vector<Segment *> createSegments(OpenCL &ocl, TubeSegmentation &TS, std::vector<CrossSection *> &crossSections, SIPL::int3 size) {
	// Create segment vector
	std::vector<Segment *> segments;

	// Do a graph component labeling
	unordered_set<int> visited;
    int labelCounter = 0;
    std::vector<std::vector<CrossSection *> > labels;
	for(int i = 0; i < crossSections.size(); i++) {
		CrossSection * c = crossSections[i];
		// Do a bfs on c
		// Check to see if point has been processed before doing a BFS
		if(visited.find(c->label) != visited.end())
			continue;

        c->label = labelCounter;
        labelCounter++;
        std::vector<CrossSection *> list;

		std::stack<CrossSection *> stack;
		stack.push(c);
		while(!stack.empty()) {
			CrossSection * current = stack.top();
			stack.pop();
			// Check label of neighbors to see if they have been added
			if(current->label != c->label || c->pos == current->pos) {
                list.push_back(current);
				// Change label of neighbors if not
				current->label = c->label;
				// Add neighbors to stack

				for(int j = 0; j < current->neighbors.size(); j++) {
					CrossSection * n = current->neighbors[j];
					if(n->label != c->label)
						stack.push(n);
				}
			}
		}
		visited.insert(c->label);
        labels.push_back(list);
	}

	std::cout << "finished graph component labeling" << std::endl;

	// Do a floyd warshall all pairs shortest path
	int totalSize = crossSections.size();
	std::cout << "number of cross sections is " << totalSize << std::endl;


    // For each label

	for(int i = 0; i < labels.size(); i++) {
		std::vector<CrossSection *> list = labels[i];
        // Do floyd warshall on all pairs
        int totalSize = list.size();
        float * dist = new float[totalSize*totalSize];
        int * pred = new int[totalSize*totalSize];

        for(int u = 0; u < totalSize; u++) {
            CrossSection * U = list[u];
            U->index = u;
        }
        #define DPOS(U, V) V+U*totalSize
	// For each cross section U
	for(int u = 0; u < totalSize; u++) {
		CrossSection * U = list[u];
		// For each cross section V
		for(int v = 0; v < totalSize; v++) {
			dist[DPOS(u,v)] = 99999999;
			pred[DPOS(u,v)] = -1;
		}
		dist[DPOS(U->index,U->index)] = 0;
		for(int j = 0; j < U->neighbors.size(); j++) {
			CrossSection * V = U->neighbors[j];
			// TODO calculate more advanced weight
			dist[DPOS(U->index,V->index)] = ceil(U->pos.distance(V->pos)) - calculateBenefit(U, V, TS, size); //(1-V->TDF);
			pred[DPOS(U->index,V->index)] = U->index;
		}
	}
	for(int t = 0; t < totalSize; t++) {
		//CrossSection * T = crossSections[t];
		//std::cout << "processing t=" << t << std::endl;
		// For each cross section U
		for(int u = 0; u < totalSize; u++) {
			//CrossSection * U = crossSections[u];
			// For each cross section V
			for(int v = 0; v < totalSize; v++) {
				//CrossSection * V = crossSections[v];
				float newLength = dist[DPOS(u, t)] + dist[DPOS(t,v)];
				if(newLength < dist[DPOS(u,v)]) {
					dist[DPOS(u,v)] = newLength;
					pred[DPOS(u,v)] = pred[DPOS(t,v)];
				}
			}
		}
}



	for(int s = 0; s < list.size(); s++) { // Source
		CrossSection * S = list[s];
		for(int t = 0; t < list.size(); t++) { // Target
			CrossSection * T = list[t];
			if(S->label == T->label && S->index != T->index) {
				Segment * segment = new Segment;
				// add all cross sections in segment
				float benefit = 0.0f;
				segment->sections.push_back(T);
				int current = T->index;
				while(current != S->index) {
					CrossSection * C = list[current];
					segment->sections.push_back(C);
					current = pred[DPOS(S->index,current)];// get predecessor
					benefit += calculateBenefit(C, list[current], TS, size);
				}
				segment->sections.push_back(list[current]);
				segment->benefit = benefit;
				segments.push_back(segment);
			}
		}
	}

        delete[] dist;
        delete[] pred;

    }
	std::cout << "finished performing floyd warshall" << std::endl;

	std::cout << "finished creating segments" << std::endl;
	std::cout << "total number of segments is " << segments.size() << std::endl;


	// Sort the segment vector on benefit
	std::sort(segments.begin(), segments.end(), segmentCompare);
	unordered_set<int> segmentation;

	// Go through sorted vector and do a region growing
	std::vector<Segment *> filteredSegments;
	int counter = 0;
	for(int i = 0; i < segments.size(); i++) {
		Segment * s = segments[i];
		if(!segmentInSegmentation(s, segmentation, size)) {
			//std::cout << "adding segment with benefit: " << s->benefit << std::endl;
			// Do region growing and Add all segmented voxels to a set
			inverseGradientRegionGrowing(s, TS, segmentation, size);
			filteredSegments.push_back(s);
			s->index = counter;
			counter++;
		}
	}

	std::cout << "total number of segments after remove overlapping segments " << filteredSegments.size() << std::endl;

	return filteredSegments;
}

int selectRoot(std::vector<Segment *> segments) {
	int root = 0;
	for(int i = 1; i < segments.size(); i++) {
		if(segments[i]->benefit > segments[root]->benefit)
			root = i;
	}
	return root;
}

void DFS(Segment * current, int * ordering, int &counter, unordered_set<int> &visited) {
	if(visited.find(current->index) != visited.end())
		return;
	ordering[counter] = current->index;
	//std::cout << counter << ": " << current->index << std::endl;
	counter++;
	for(int i = 0; i < current->connections.size(); i++) {
		Connection * edge = current->connections[i];
		DFS(edge->target, ordering, counter, visited);
	}
	visited.insert(current->index);
}

int * createDepthFirstOrdering(std::vector<Segment *> segments, int root, int &Ns) {
	int * ordering = new int[segments.size()];
	int counter = 0;

	// Give imdexes to segments
	for(int i = 0; i < segments.size(); i++) {
		segments[i]->index = i;
	}

	unordered_set<int> visited;

	DFS(segments[root], ordering, counter, visited);

	Ns = counter;
	int * reversedOrdering = new int[Ns];
	for(int i = 0; i < Ns; i++) {
		reversedOrdering[i] = ordering[Ns-i-1];
	}

	delete[] ordering;
	return reversedOrdering;
}

class ConnectionComparator {
public:
	bool operator()(Connection * a, Connection *b) const {
		return a->cost > b->cost;
	}
};

std::vector<Segment *> minimumSpanningTree(Segment * root, int3 size) {
	// Need a priority queue on Connection objects based on the cost
	std::priority_queue<Connection *, std::vector<Connection *>, ConnectionComparator> queue;
	std::vector<Segment *> result;
	unordered_set<int> visited;
	result.push_back(root);
	visited.insert(root->index);

	// Add all connections of the root to the queue
	for(int i = 0; i < root->connections.size(); i++) {
		Connection * c = root->connections[i];
		queue.push(c);
	}
	// Remove connections from root
	root->connections = std::vector<Connection *>();

	while(!queue.empty()) {
	// Select minimum connection
	// Check if target is already added
	// if not, add all of its connection to the queue
	// add this connection to the source
	// Add target segment and clear its connections
	// Also add cost to the segment object
		Connection * c = queue.top();
		//std::cout << c->cost << std::endl;
		queue.pop();
		if(visited.find(c->target->index) != visited.end())
			continue;

		for(int i = 0; i < c->target->connections.size(); i++) {
			Connection * cn = c->target->connections[i];
			if(visited.find(cn->target->index) == visited.end())
				queue.push(cn);
		}

		c->source->connections.push_back(c);
		// c->target->connections.clear(); doest his delete the objects?
		c->target->connections = std::vector<Connection *>();
		c->target->cost = c->cost;
		result.push_back(c->target);
		visited.insert(c->target->index);
	}

	return result;
}

std::vector<Segment *> findOptimalSubtree(std::vector<Segment *> segments, int * depthFirstOrdering, int Ns) {

	float * score = new float[Ns]();
	float r = 2.0;

	// Stage 1 bottom up
	for(int j = 0; j < Ns; j++) {
		int mj = depthFirstOrdering[j];
		score[mj] = segments[mj]->benefit - r * segments[mj]->cost;
		/*std::cout << "cross sections: " << segments[mj]->sections.size() << " benefit: "
				<< segments[mj]->benefit << " cost: " << segments[mj]->cost <<
				" children: " << segments[mj]->connections.size() << std::endl;*/
		// For all children of mj
		for(int n = 0; n < segments[mj]->connections.size(); n++) {
			Connection * c = segments[mj]->connections[n];
			int k = c->target->index; //child
			if(score[k] >= 0)
				score[mj] += score[k];
		}
	}

	// Stage 2 top down
	bool * v = new bool[Ns];
	for(int i = 1; i < Ns; i++)
		v[i] = false;
	v[0] = true;

	for(int j = Ns-1; j >= 0; j--) {
		int mj = depthFirstOrdering[j];
		if(v[mj]) {
			// For all children of mj
			for(int n = 0; n < segments[mj]->connections.size(); n++) {
				Connection * c = segments[mj]->connections[n];
				int k = c->target->index; //child
				if(score[k] >= 0)
					v[k] = true;
			}
		}
	}

	delete[] score;

	std::vector<Segment *> finalSegments;
	for(int i = 0; i < Ns; i++) {
		if(v[i]) {
			finalSegments.push_back(segments[i]);

			// for all children, check if they are true in v, if not remove connections
			std::vector<Connection *> connections;
			for(int n = 0; n < segments[i]->connections.size(); n++) {
				Connection * c = segments[i]->connections[n];
				int k = c->target->index; //child
				if(v[k]) {
					// keep connection
					connections.push_back(c);
				} else {
					delete c;
				}
			}
			segments[i]->connections = connections;
		}
	}
	delete[] v;
	return finalSegments;
}

float calculateConnectionCost(CrossSection * a, CrossSection * b, TubeSegmentation &TS, int3 size) {
	float cost = 0.0f;
	int distance = ceil(a->pos.distance(b->pos));
	float3 direction(b->pos.x-a->pos.x,b->pos.y-a->pos.y,b->pos.z-a->pos.z);
	float maxIntensity = -1.0f;
	for(int i = 0; i < distance; i++) {
		float frac = (float)i/distance;
		float3 n = a->pos + frac*direction;
		int3 in(round(n.x),round(n.y),round(n.z));
		cost += /*SQR_MAG(in) +*/ (1.0f-TS.TDF[POS(in)]);
        //float3 e1 = getTubeDirection(TS, in, size);
        //cost += (1-fabs(a->direction.dot(e1)))+(1-fabs(b->direction.dot(e1)));
		if(TS.intensity[POS(in)] > maxIntensity) {
			maxIntensity = TS.intensity[POS(in)];
		}
	}
	/*
	if(maxIntensity > 0.2 && maxIntensity < 0.3) {
		cost = cost*2;
	}
	if(maxIntensity >= 0.3 && maxIntensity < 0.5) {
		cost = cost*4;
	}
	if(maxIntensity >= 0.5) {
		cost = cost*8;
	}
	*/
	return cost;
}

void createConnections(TubeSegmentation &TS, std::vector<Segment *> segments, int3 size) {
	// For all pairs of segments
	for(int k = 0; k < segments.size(); k++) {
		Segment * s_k = segments[k];
		for(int l = 0; l < k; l++) {
			Segment * s_l = segments[l];
			// For each C_k, cross sections in S_k, calculate costs and select the one with least cost
			float bestCost = 999999999.0f;
			CrossSection * c_k_best, * c_l_best;
			bool found = false;
			for(int i = 0; i < s_k->sections.size(); i++){
				CrossSection * c_k = s_k->sections[i];
				for(int j = 0; j < s_l->sections.size(); j++){
					CrossSection * c_l = s_l->sections[j];
					if(c_k->pos.distance(c_l->pos) > 20)
						continue;

					float3 c(c_k->pos.x-c_l->pos.x, c_k->pos.y-c_l->pos.y,c_k->pos.z-c_l->pos.z);
					c = c.normalize();
					if(acos(fabs(c_k->direction.dot(c))) > 1.05f)
						continue;
					if(acos(fabs(c_l->direction.dot(c))) > 1.05f)
						continue;

					/*
                    float rk = TS.radius[POS(c_k->pos)];
                    float rl = TS.radius[POS(c_l->pos)];

                    if(rk > 2 || rl > 2) {
                        if(std::max(rk,rl) / std::min(rk,rl) >= 2)
                            continue;
                    }
                    */

					float cost = calculateConnectionCost(c_k, c_l, TS, size);
					if(cost < bestCost) {
						bestCost = cost;
						c_k_best = c_k;
						c_l_best = c_l;
						found = true;
					}
				}
			}


			// See if they are allowed to connect
			if(found) {
				/*if(bestCost < 2) {
					std::cout << bestCost << std::endl;
					std::cout << "labels: " << c_k_best->label << " " << c_l_best->label << std::endl;
					std::cout << "distance: " << c_k_best->pos.distance(c_l_best->pos) << std::endl;
				}*/
				// If so, create connection object and add to segemnt
				Connection * c = new Connection;
				c->cost = bestCost;
				c->source = s_k;
				c->source_section = c_k_best;
				c->target = s_l;
				c->target_section = c_l_best;
				s_k->connections.push_back(c);
				Connection * c2 = new Connection;
				c2->cost = bestCost;
				c2->source = s_l;
				c2->source_section = c_l_best;
				c2->target = s_k;
				c2->target_section = c_k_best;
				s_l->connections.push_back(c2);
			}
		}
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

void TSFGarbageCollector::addMemObject(cl::Memory* mem) {
    memObjects.insert(mem);
}

void TSFGarbageCollector::deleteMemObject(cl::Memory* mem) {
    memObjects.erase(mem);
    delete mem;
    mem = NULL;
}

void TSFGarbageCollector::deleteAllMemObjects() {
    std::set<cl::Memory *>::iterator it;
    for(it = memObjects.begin(); it != memObjects.end(); it++) {
        cl::Memory * mem = *it;
        delete (mem);
        mem = NULL;
    }
    memObjects.clear();
}

TSFGarbageCollector::~TSFGarbageCollector() {
    deleteAllMemObjects();
}
