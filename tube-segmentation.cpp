#include "tube-segmentation.hpp"
#include "SIPL/Types.hpp"
#include "SIPL/Core.hpp"
#include <boost/iostreams/device/mapped_file.hpp>
#include <queue>
#include <stack>
#include <list>
#include <cstdio>
#include <limits>
#ifdef CPP11
#include <unordered_set>
using std::unordered_set;
#else
#include <boost/unordered_set.hpp>
using boost::unordered_set;
#endif
#include "histogram-pyramids.hpp"
#include "tsf-config.h"

//#define TIMING

#ifdef TIMING
#include <chrono>
#define INIT_TIMER auto timerStart = std::chrono::high_resolution_clock::now();
#define START_TIMER  timerStart = std::chrono::high_resolution_clock::now();
#define STOP_TIMER(name)  std::cout << "RUNTIME of " << name << ": " << \
        std::chrono::duration_cast<std::chrono::milliseconds>( \
                            std::chrono::high_resolution_clock::now()-timerStart \
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


TSFOutput * run(std::string filename, paramList &parameters) {

    INIT_TIMER
    OpenCL * ocl = new OpenCL;
    cl_device_type type;
    if(parameters.strings["device"].get() == "gpu") {
    	type = CL_DEVICE_TYPE_GPU;
    } else {
    	type = CL_DEVICE_TYPE_CPU;
    }
	ocl->context = createCLContext(type);

    // Select first device
    cl::vector<cl::Device> devices = ocl->context.getInfo<CL_CONTEXT_DEVICES>();
    std::cout << "Using device: " << devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;
    ocl->queue = cl::CommandQueue(ocl->context, devices[0], CL_QUEUE_PROFILING_ENABLE);

    // Query the size of available memory
    unsigned int memorySize = devices[0].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    std::cout << "Available memory on selected device " << (double)memorySize/(1024*1024) << " MB "<< std::endl;

    // Compile and create program
    if(!getParamBool(parameters, "buffers-only") && (int)devices[0].getInfo<CL_DEVICE_EXTENSIONS>().find("cl_khr_3d_image_writes") > -1) {
    	std::string filename = std::string(KERNELS_DIR)+"/kernels.cl";
        ocl->program = buildProgramFromSource(ocl->context, filename.c_str());
        BoolParameter v = parameters.bools["3d_write"];
        v.set(true);
        parameters.bools["3d_write"] = v;
    } else {
        BoolParameter v = parameters.bools["3d_write"];
        v.set(false);
        parameters.bools["3d_write"] = v;
        std::string filename = std::string(KERNELS_DIR)+"/kernels_no_3d_write.cl";
        ocl->program = buildProgramFromSource(ocl->context, filename.c_str());
        std::cout << "NOTE: Writing to 3D textures is not supported on the selected device." << std::endl;
    }

    START_TIMER
    SIPL::int3 * size = new SIPL::int3();
    TSFOutput * output = new TSFOutput(ocl, size);
    try {
        // Read dataset and transfer to device
        cl::Image3D dataset = readDatasetAndTransfer(*ocl, filename, parameters, size, output);

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
        throw SIPL::SIPLException(str.c_str());
    }
    ocl->queue.finish();
    STOP_TIMER("total")
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
                            if(T.TDF[POS(nPos)] > T.TDF[POS(pos)]) {
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
    if(maskSize < 1) // cap min mask size at 3x3
    	maskSize = 1;
    if(maskSize > 8) // cap mask size at 17x17
    	maskSize = 8;
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

void runCircleFittingMethod(OpenCL &ocl, Image3D &dataset, SIPL::int3 size, paramList &parameters, Image3D &vectorField, Image3D &TDF, Image3D &radiusImage) {
    // Set up parameters
    const int GVFIterations = getParam(parameters, "gvf-iterations");
    const float radiusMin = getParam(parameters, "radius-min");
    const float radiusMax = getParam(parameters, "radius-max");
    const float radiusStep = getParam(parameters, "radius-step");
    const float Fmax = getParam(parameters, "fmax");
    const int totalSize = size.x*size.y*size.z;
    const bool no3Dwrite = !getParamBool(parameters, "3d_write");
    const float MU = getParam(parameters, "gvf-mu");
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

    Image3D blurredVolume = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_R, CL_FLOAT), size.x, size.y, size.z);
    Buffer TDFsmall;
    Buffer radiusSmall;
    if(radiusMin < 2.5f) {
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
			blurVolumeWithGaussianKernel.setArg(0, dataset);
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
					blurredVolume,
					0,
					offset,
					region
			);
    	} else {
			// Run blurVolumeWithGaussian on processedVolume
			blurVolumeWithGaussianKernel.setArg(0, dataset);
			blurVolumeWithGaussianKernel.setArg(1, blurredVolume);
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
    if(no3Dwrite) {
        // Create auxillary buffer
        Buffer vectorFieldBuffer = Buffer(ocl.context, CL_MEM_WRITE_ONLY, 4*sizeof(float)*totalSize);
        vectorField = Image3D(ocl.context, CL_MEM_READ_ONLY, ImageFormat(CL_RGBA, CL_FLOAT), size.x, size.y, size.z);
 
        // Run create vector field
        createVectorFieldKernel.setArg(0, blurredVolume);
        createVectorFieldKernel.setArg(1, vectorFieldBuffer);
        createVectorFieldKernel.setArg(2, Fmax);
        createVectorFieldKernel.setArg(3, vectorSign);

        ocl.queue.enqueueNDRangeKernel(
                createVectorFieldKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NullRange
        );

        // Copy buffer contents to image
        ocl.queue.enqueueCopyBufferToImage(
                vectorFieldBuffer, 
                vectorField, 
                0,
                offset,
                region
        );

    } else {
        if(getParamBool(parameters, "32bit-vectors")) {
            std::cout << "NOTE: Using 32 bit vectors" << std::endl;
            vectorField = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_RGBA, CL_FLOAT), size.x, size.y, size.z);
        } else {
            std::cout << "NOTE: Using 16 bit vectors" << std::endl;
            vectorField = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_RGBA, CL_SNORM_INT16), size.x, size.y, size.z);
        }
     
        // Run create vector field
        createVectorFieldKernel.setArg(0, blurredVolume);
        createVectorFieldKernel.setArg(1, vectorField);
        createVectorFieldKernel.setArg(2, Fmax);
        createVectorFieldKernel.setArg(3, vectorSign);

        ocl.queue.enqueueNDRangeKernel(
                createVectorFieldKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NDRange(4,4,4)
        );

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
    TDFsmall = Buffer(ocl.context, CL_MEM_WRITE_ONLY, sizeof(float)*totalSize);
    radiusSmall = Buffer(ocl.context, CL_MEM_WRITE_ONLY, sizeof(float)*totalSize);
    circleFittingTDFKernel.setArg(0, vectorField);
    circleFittingTDFKernel.setArg(1, TDFsmall);
    circleFittingTDFKernel.setArg(2, radiusSmall);
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
		TDF = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_R, CL_FLOAT),
				size.x, size.y, size.z);
		ocl.queue.enqueueCopyBufferToImage(
			TDFsmall,
			TDF,
			0,
			offset,
			region
		);
		radiusImage = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_R, CL_FLOAT),
				size.x, size.y, size.z);
		ocl.queue.enqueueCopyBufferToImage(
			radiusSmall,
			radiusImage,
			0,
			offset,
			region
		);
		return;
    }
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
			blurVolumeWithGaussianKernel.setArg(0, dataset);
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
					blurredVolume,
					0,
					offset,
					region
			);
    	} else {
			// Run blurVolumeWithGaussian on processedVolume
			blurVolumeWithGaussianKernel.setArg(0, dataset);
			blurVolumeWithGaussianKernel.setArg(1, blurredVolume);
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
    ocl.queue.enqueueMarker(&endEvent);
    ocl.queue.finish();
    startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    endEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &end);
    std::cout << "RUNTIME blurring: " << (end-start)*1.0e-6 << " ms" << std::endl;
}
if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&startEvent);
}
   if(no3Dwrite) {
        // Create auxillary buffer
        Buffer vectorFieldBuffer = Buffer(ocl.context, CL_MEM_WRITE_ONLY, 4*sizeof(float)*totalSize);
        vectorField = Image3D(ocl.context, CL_MEM_READ_ONLY, ImageFormat(CL_RGBA, CL_FLOAT), size.x, size.y, size.z);
 
        // Run create vector field
        createVectorFieldKernel.setArg(0, blurredVolume);
        createVectorFieldKernel.setArg(1, vectorFieldBuffer);
        createVectorFieldKernel.setArg(2, Fmax);
        createVectorFieldKernel.setArg(3, vectorSign);

        ocl.queue.enqueueNDRangeKernel(
                createVectorFieldKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NullRange
        );

        // Copy buffer contents to image
        ocl.queue.enqueueCopyBufferToImage(
                vectorFieldBuffer, 
                vectorField, 
                0,
                offset,
                region
        );

    } else {
        if(getParamBool(parameters, "32bit-vectors")) {
            vectorField = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_RGBA, CL_FLOAT), size.x, size.y, size.z);
        } else {
            vectorField = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_RGBA, CL_SNORM_INT16), size.x, size.y, size.z);
        }

     
        // Run create vector field
        createVectorFieldKernel.setArg(0, blurredVolume);
        createVectorFieldKernel.setArg(1, vectorField);
        createVectorFieldKernel.setArg(2, Fmax);
        createVectorFieldKernel.setArg(3, vectorSign);

        ocl.queue.enqueueNDRangeKernel(
                createVectorFieldKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NDRange(4,4,4)
        );

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

    // Run GVF on iVectorField as initial vector field
    Kernel GVFInitKernel = Kernel(ocl.program, "GVF3DInit");
    Kernel GVFIterationKernel = Kernel(ocl.program, "GVF3DIteration");
    Kernel GVFFinishKernel = Kernel(ocl.program, "GVF3DFinish");

    std::cout << "Running GVF with " << GVFIterations << " iterations " << std::endl; 
    if(no3Dwrite) {
        // Create auxillary buffers
        Buffer vectorFieldBuffer = Buffer(
                ocl.context,
                CL_MEM_READ_WRITE,
                3*sizeof(float)*totalSize
        );
        Buffer vectorFieldBuffer1 = Buffer(
                ocl.context,
                CL_MEM_READ_WRITE,
                3*sizeof(float)*totalSize
        );

        GVFInitKernel.setArg(0, vectorField);
        GVFInitKernel.setArg(1, vectorFieldBuffer);
        ocl.queue.enqueueNDRangeKernel(
                GVFInitKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NullRange
        );

        // Run iterations
        GVFIterationKernel.setArg(0, vectorField);
        GVFIterationKernel.setArg(3, MU);

        for(int i = 0; i < GVFIterations; i++) {
            if(i % 2 == 0) {
                GVFIterationKernel.setArg(1, vectorFieldBuffer);
                GVFIterationKernel.setArg(2, vectorFieldBuffer1);
            } else {
                GVFIterationKernel.setArg(1, vectorFieldBuffer1);
                GVFIterationKernel.setArg(2, vectorFieldBuffer);
            }
                ocl.queue.enqueueNDRangeKernel(
                        GVFIterationKernel,
                        NullRange,
                        NDRange(size.x,size.y,size.z),
                        NullRange
                );
        }

        vectorFieldBuffer1 = Buffer(
                ocl.context,
                CL_MEM_WRITE_ONLY,
                4*sizeof(float)*totalSize
        );

        // Copy vector field to image
        GVFFinishKernel.setArg(0, vectorFieldBuffer);
        GVFFinishKernel.setArg(1, vectorFieldBuffer1);

        ocl.queue.enqueueNDRangeKernel(
                GVFFinishKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NullRange
        );

        // Copy buffer contents to image
        ocl.queue.enqueueCopyBufferToImage(
                vectorFieldBuffer1, 
                vectorField, 
                0,
                offset,
                region
        );


    } else {
        Image3D vectorField1;
        Image3D initVectorField; 
        if(getParamBool(parameters, "32bit-vectors")) {
            vectorField1 = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_RGBA, CL_FLOAT), size.x, size.y, size.z);
            initVectorField = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_RG, CL_FLOAT), size.x, size.y, size.z);
        } else {
            vectorField1 = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_RGBA, CL_SNORM_INT16), size.x, size.y, size.z);
            initVectorField = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_RG, CL_SNORM_INT16), size.x, size.y, size.z);
        }

        // init vectorField from image
        GVFInitKernel.setArg(0, vectorField);
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
                GVFIterationKernel.setArg(2, vectorField);
            } else {
                GVFIterationKernel.setArg(1, vectorField);
                GVFIterationKernel.setArg(2, vectorField1);
            }
                ocl.queue.enqueueNDRangeKernel(
                        GVFIterationKernel,
                        NullRange,
                        NDRange(size.x,size.y,size.z),
                        NDRange(4,4,4)
                );
        }

        // Copy vector field to image
        GVFFinishKernel.setArg(0, vectorField1);
        GVFFinishKernel.setArg(1, vectorField);

        ocl.queue.enqueueNDRangeKernel(
                GVFFinishKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NDRange(4,4,4)
        );
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
    Buffer TDFlarge = Buffer(ocl.context, CL_MEM_WRITE_ONLY, sizeof(float)*totalSize);
    Buffer radiusLarge = Buffer(ocl.context, CL_MEM_WRITE_ONLY, sizeof(float)*totalSize);
    circleFittingTDFKernel.setArg(0, vectorField);
    circleFittingTDFKernel.setArg(1, TDFlarge);
    circleFittingTDFKernel.setArg(2, radiusLarge);
    circleFittingTDFKernel.setArg(3, std::max(1.0f, radiusMin));
    circleFittingTDFKernel.setArg(4, radiusMax);
    circleFittingTDFKernel.setArg(5, radiusStep);

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
		combineKernel.setArg(0, TDFsmall);
		combineKernel.setArg(1, radiusSmall);
		combineKernel.setArg(2, TDFlarge);
		combineKernel.setArg(3, radiusLarge);

		ocl.queue.enqueueNDRangeKernel(
				combineKernel,
				NullRange,
				NDRange(totalSize),
				NDRange(64)
		);
	}
    TDF = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_R, CL_FLOAT),
            size.x, size.y, size.z);
    ocl.queue.enqueueCopyBufferToImage(
        TDFlarge,
        TDF,
        0,
        offset,
        region
    );
    radiusImage = Image3D(ocl.context, CL_MEM_READ_WRITE, ImageFormat(CL_R, CL_FLOAT),
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

Image3D runInverseGradientSegmentation(OpenCL &ocl, Image3D &centerline, Image3D &vectorField, SIPL::int3 size, paramList parameters) {
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

Image3D runNewCenterlineAlg(OpenCL &ocl, SIPL::int3 size, paramList &parameters, Image3D &vectorField, Image3D &TDF, Image3D &radius, Image3D &intensity) {
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
    Image3D centerpointsImage2 = Image3D(
            ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_R, CL_SIGNED_INT8),
            size.x, size.y, size.z
    );
    Buffer vertices;
    int sum = 0;

    if(no3Dwrite) {
        int hpSize;
        if(size.x == size.y && size.y == size.z && log2(size.x) == round(log2(size.x))) {
            hpSize = size.x;
        }else{
            // Find largest size and find closest power of two
            int largestSize = std::max(size.x, std::max(size.y, size.z));
            int i = 1;
            while(pow(2.0, i) < largestSize)
                i++;
            hpSize = pow(2.0, i);
        }


        Buffer centerpoints = Buffer(
                ocl.context,
                CL_MEM_READ_WRITE,
                sizeof(char)*hpSize*hpSize*hpSize
        );

        initCharBuffer.setArg(0, centerpoints);
        ocl.queue.enqueueNDRangeKernel(
                initCharBuffer,
                NullRange,
                NDRange(hpSize*hpSize*hpSize),
                NullRange
        );

        candidatesKernel.setArg(0, TDF);
        candidatesKernel.setArg(1, centerpoints);
        candidatesKernel.setArg(2, Thigh);
        ocl.queue.enqueueNDRangeKernel(
                candidatesKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NullRange
        );

        HistogramPyramid3DBuffer hp3(ocl);
        hp3.create(centerpoints, hpSize, hpSize, hpSize);

        candidates2Kernel.setArg(0, TDF);
        candidates2Kernel.setArg(1, radius);
        candidates2Kernel.setArg(2, vectorField);
        Buffer centerpoints2 = Buffer(
                ocl.context,
                CL_MEM_READ_WRITE,
                sizeof(char)*totalSize
        );
        initCharBuffer.setArg(0, centerpoints2);
        ocl.queue.enqueueNDRangeKernel(
                initCharBuffer,
                NullRange,
                NDRange(totalSize),
                NullRange
        );

        candidates2Kernel.setArg(3, centerpoints2);
        std::cout << "candidates: " << hp3.getSum() << std::endl;
        hp3.traverse(candidates2Kernel, 4);
        ocl.queue.enqueueCopyBufferToImage(
            centerpoints2,
            centerpointsImage2,
            0,
            offset,
            region
        );

		if(getParamBool(parameters, "centerpoints-only")) {
			return centerpointsImage2;
		}
        ddKernel.setArg(0, vectorField);
        ddKernel.setArg(1, TDF);
        ddKernel.setArg(2, centerpointsImage2);
        ddKernel.setArg(4, cubeSize);
        Buffer centerpoints3 = Buffer(
                ocl.context,
                CL_MEM_READ_WRITE,
                sizeof(char)*hpSize*hpSize*hpSize
        );
        initCharBuffer.setArg(0, centerpoints3);
        ocl.queue.enqueueNDRangeKernel(
                initCharBuffer,
                NullRange,
                NDRange(hpSize*hpSize*hpSize),
                NullRange
        );
        ddKernel.setArg(3, centerpoints3);
        ocl.queue.enqueueNDRangeKernel(
                ddKernel,
                NullRange,
                NDRange(ceil((float)size.x/cubeSize),ceil((float)size.y/cubeSize),ceil((float)size.z/cubeSize)),
                NullRange
        );

        // Construct HP of centerpointsImage
        HistogramPyramid3DBuffer hp(ocl);
        hp.create(centerpoints3, hpSize, hpSize, hpSize);
        sum = hp.getSum();
        std::cout << "number of vertices detected " << sum << std::endl;

        // Run createPositions kernel
        vertices = hp.createPositionBuffer(); 
    } else {
        Kernel init3DImage(ocl.program, "init3DImage");
        init3DImage.setArg(0, centerpointsImage2);
        ocl.queue.enqueueNDRangeKernel(
            init3DImage,
            NullRange,
            NDRange(size.x,size.y,size.z),
            NullRange
        );

        Image3D centerpointsImage = Image3D(
                ocl.context,
                CL_MEM_READ_WRITE,
                ImageFormat(CL_R, CL_SIGNED_INT8),
                size.x, size.y, size.z
        );

        candidatesKernel.setArg(0, TDF);
        candidatesKernel.setArg(1, centerpointsImage);
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
        hp3.create(centerpointsImage, size.x, size.y, size.z);
        std::cout << "candidates: " << hp3.getSum() << std::endl;
        candidates2Kernel.setArg(3, centerpointsImage2);
        hp3.traverse(candidates2Kernel, 4);

        Image3D centerpointsImage3 = Image3D(
                ocl.context,
                CL_MEM_READ_WRITE,
                ImageFormat(CL_R, CL_SIGNED_INT8),
                size.x, size.y, size.z
        );
        init3DImage.setArg(0, centerpointsImage3);
        ocl.queue.enqueueNDRangeKernel(
            init3DImage,
            NullRange,
            NDRange(size.x,size.y,size.z),
            NullRange
        );

		if(getParamBool(parameters, "centerpoints-only")) {
			return centerpointsImage2;
		}
        ddKernel.setArg(0, vectorField);
        ddKernel.setArg(1, TDF);
        ddKernel.setArg(2, centerpointsImage2);
        ddKernel.setArg(4, cubeSize);
        ddKernel.setArg(3, centerpointsImage3);
        ocl.queue.enqueueNDRangeKernel(
                ddKernel,
                NullRange,
                NDRange(ceil((float)size.x/cubeSize),ceil((float)size.y/cubeSize),ceil((float)size.z/cubeSize)),
                NullRange
        );

        // Construct HP of centerpointsImage
        HistogramPyramid3D hp(ocl);
        hp.create(centerpointsImage3, size.x, size.y, size.z);
        sum = hp.getSum();
        std::cout << "number of vertices detected " << sum << std::endl;

        // Run createPositions kernel
        vertices = hp.createPositionBuffer(); 

    }
    if(sum < 8 || sum >= 16384) {
    	throw SIPL::SIPLException("Too many or too few vertices detected", __LINE__, __FILE__);
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
    Image2D lengths = Image2D(
            ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_R, CL_FLOAT),
            sum, sum
    );

    // Run linkLengths kernel
    Kernel linkLengths(ocl.program, "linkLengths");
    linkLengths.setArg(0, vertices);
    linkLengths.setArg(1, lengths);
    ocl.queue.enqueueNDRangeKernel(
            linkLengths,
            NullRange,
            NDRange(sum, sum),
            NullRange
    );

    // Create and init compacted_lengths image
    float * cl = new float[sum*sum*2]();
    Image2D compacted_lengths = Image2D(
            ocl.context,
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            ImageFormat(CL_RG, CL_FLOAT),
            sum, sum,
            0,
            cl
    );

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
    compactLengths.setArg(0, lengths);
    compactLengths.setArg(1, incs);
    compactLengths.setArg(2, compacted_lengths);
    compactLengths.setArg(3, maxDistance);
    ocl.queue.enqueueNDRangeKernel(
            compactLengths,
            NullRange,
            NDRange(sum, sum),
            NullRange
    );

    Kernel linkingKernel(ocl.program, "linkCenterpoints");
    linkingKernel.setArg(0, TDF);
    linkingKernel.setArg(1, radius);
    linkingKernel.setArg(2, vertices);
    linkingKernel.setArg(3, edgeTuples);
    linkingKernel.setArg(4, intensity);
    linkingKernel.setArg(5, compacted_lengths);
    linkingKernel.setArg(6, sum);
    linkingKernel.setArg(7, Tmean);
    linkingKernel.setArg(8, maxDistance);
    ocl.queue.enqueueNDRangeKernel(
            linkingKernel,
            NullRange,
            NDRange(globalSize),
            NDRange(64)
    );
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
    Image3D centerlines;    
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

        centerlines = Image3D(
            ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_R, CL_SIGNED_INT8),
            size.x, size.y, size.z
        );

        ocl.queue.enqueueCopyBufferToImage(
                centerlinesBuffer,
                centerlines,
                0,
                offset,
                region
        );

    } else {
        centerlines = Image3D(
            ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_R, CL_SIGNED_INT8),
            size.x, size.y, size.z
        );

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

    if(getParamStr(parameters, "centerline-vtk-file") != "off") {
    	// Transfer edges (size: sum2) and vertices (size: sum) buffers to host
    	int * verticesArray = new int[sum*3];
    	int * edgesArray = new int[sum2*3];

    	ocl.queue.enqueueReadBuffer(vertices, CL_FALSE, 0, sum*3*sizeof(int), verticesArray);
    	ocl.queue.enqueueReadBuffer(edges, CL_FALSE, 0, sum2*2*sizeof(int), edgesArray);

    	ocl.queue.finish();


    	// Write to file
    	std::ofstream file;
    	file.open(getParamStr(parameters, "centerline-vtk-file").c_str());
    	file << "# vtk DataFile Version 3.0\nvtk output\nASCII\n";
    	file << "DATASET POLYDATA\nPOINTS " << sum << " int\n";
    	for(int i = 0; i < sum; i++) {
    		file << verticesArray[i*3] << " " << verticesArray[i*3+1] << " " << verticesArray[i*3+2] << "\n";
    	}

    	file << "\nLINES " << sum2 << " " << sum2*3 << "\n";
    	for(int i = 0; i < sum2; i++) {
    		file << "2 " << edgesArray[i*2] << " " << edgesArray[i*2+1] << "\n";
    	}

    	file.close();
    	delete[] verticesArray;
    	delete[] edgesArray;
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

void writeDataToDisk(TSFOutput * output, std::string storageDirectory) {
	SIPL::int3 * size = output->getSize();
	if(output->hasCenterlineVoxels())
		writeToRaw<char>(output->getCenterlineVoxels(), storageDirectory + "centerline.raw", size->x, size->y, size->z);

	if(output->hasSegmentation())
		writeToRaw<char>(output->getSegmentation(), storageDirectory + "segmentation.raw", size->x, size->y, size->z);
}

void runCircleFittingAndNewCenterlineAlg(OpenCL * ocl, cl::Image3D &dataset, SIPL::int3 * size, paramList &parameters, TSFOutput * output) {
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

    Image3D * centerline = new Image3D;
    *centerline = runNewCenterlineAlg(*ocl, *size, parameters, vectorField, *TDF, radius, dataset);
    output->setCenterlineVoxels(centerline);

    Image3D * segmentation = new Image3D;
    if(!getParamBool(parameters, "no-segmentation")) {
    	if(!getParamBool(parameters, "sphere-segmentation")) {
			*segmentation = runInverseGradientSegmentation(*ocl, *centerline, vectorField, *size, parameters);
    	} else {
			*segmentation = runSphereSegmentation(*ocl, *centerline, radius, *size, parameters);
    	}
    	output->setSegmentation(segmentation);
    }

	if(getParamStr(parameters, "storage-dir") != "off") {
		writeDataToDisk(output, getParamStr(parameters, "storage-dir"));
    }

}

class CrossSection {
public:
	int3 pos;
	float TDF;
	std::vector<CrossSection *> neighbors;
	int label;
};

std::vector<CrossSection *> createGraph(TubeSegmentation &TS, SIPL::int3 size) {
	// Create vector
	std::vector<CrossSection *> sections;
	float threshold = 0.5f;

	// Go through TS.TDF and add all with TDF above threshold
	int counter = 0;
	for(int z = 1; z < size.z-1; z++) {
	for(int y = 1; y < size.y-1; y++) {
	for(int x = 1; x < size.x-1; x++) {
		int3 pos(x,y,z);
		float tdf = TS.TDF[POS(pos)];
		if(tdf > threshold) {
			CrossSection * cs = new CrossSection;
			cs->pos = pos;
			cs->TDF = tdf;
			cs->label = counter;
			counter++;
			sections.push_back(cs);
		}
	}}}

	std::vector<CrossSection *> sectionPairs;

	// For each cross section c_i
	for(CrossSection * c_i : sections) {
		// For each cross section c_j
		for(CrossSection * c_j : sections) {
			// If all criterias are ok: Add c_j as neighbor to c_i
			if(c_i->pos.distance(c_j->pos) < 5) {
				float3 e1_i = getTubeDirection(TS, c_i->pos, size);
				float3 e1_j = getTubeDirection(TS, c_j->pos, size);
				int3 cint = c_i->pos - c_j->pos;
				float3 c = cint.normalize();

				if(acos((double)fabs(e1_i.dot(e1_j))) > 1.05) // 60 degrees
					continue;

				if(acos((double)fabs(e1_i.dot(c))) > 1.05)
					continue;

				if(acos((double)fabs(e1_j.dot(c))) > 1.05)
					continue;

				c_i->neighbors.push_back(c_j);
				sectionPairs.push_back(c_i);
			}
			// If no pair is found, dont add it
		}
	}

	return sectionPairs;
}

class CrossSectionComparator {
private:
	unordered_map<int, float> distance;
	int3 size;
public:
	CrossSectionComparator(unordered_map<int, float> &distance, SIPL::int3 size) { this->distance = distance; this->size = size;};
bool operator() (const CrossSection * lhs, const CrossSection * rhs) {
	float a = distance[POS(lhs->pos)];
	float b =  distance[POS(rhs->pos)];
	return a > b;
};
};

class Segment {
public:
	std::vector<int3> crossSections;
	float benefit;
};

std::vector<Segment *> createSegments(TubeSegmentation &TS, std::vector<CrossSection *> &crossSections, SIPL::int3 size) {
	// Create segment vector
	std::vector<Segment *> segments;

	// Do a graph component labeling
	unordered_set<int> visited;
	for(CrossSection * c : crossSections) {
		// Do a bfs on c
		// Check to see if point has been processed before doing a BFS
		if(visited.find(c->label) != visited.end())
			continue;

		std::stack<CrossSection *> stack;
		stack.push(c);
		while(!stack.empty()) {
			CrossSection * current = stack.top();
			stack.pop();
			// Check label of neighbors to see if they have been added
			if(current->label != c->label || c->pos == current->pos) {
				// Change label of neighbors if not
				current->label = c->label;
				// Add neighbors to stack
				for(CrossSection * n : current->neighbors) {
					if(n->label != c->label)
						stack.push(n);
				}
			}
		}
		visited.insert(c->label);
	}


	// For each cross section c_i
	for(CrossSection * c_i : crossSections) {
		// For each cross section c_j
		for(CrossSection * c_j : crossSections) {
			// If they have the same label
			if(c_i->label == c_j->label) {
				// Do a djikstra on the tdf to find best segment between i and j

				unordered_map<int, float> distance;
				for(CrossSection * c : crossSections)
					distance[POS(c->pos)] = 9999999;
				unordered_set<int> visited;
				std::priority_queue<CrossSection *, std::vector<CrossSection *>, CrossSectionComparator> queue(CrossSectionComparator(distance, size), std::vector<CrossSection *>());
				distance[POS(c_i->pos)] = 0;
				queue.push(c_i);
				Segment * segment = new Segment;

				while(!queue.empty()) {
					CrossSection * c = queue.top();
					queue.pop();
					if(visited.find(POS(c->pos)) != visited.end())
						continue;

					if(c->pos == c_j->pos) // end found
						break;

					for(CrossSection * n : c->neighbors) {
						// TODO calculate weights from c to n
						float weight = 1-n->TDF;
						float newDistance = distance[POS(c->pos)] + weight;
						if(newDistance < distance[POS(n->pos)]) {
							distance[POS(n->pos)] = newDistance;
							queue.push(n);
						}
					}
					visited.insert(POS(c->pos));
				}

				// TODO set cross sections of segment
				// TODO set benefit of segment

				// Add segment to vector
				segments.push_back(segment);
			}
		}
	}

	// Sort the segment vector on benefit
	// Go through sorted vector and do a region growing

	return segments;
}

void runCircleFittingAndTest(OpenCL * ocl, cl::Image3D &dataset, SIPL::int3 * size, paramList &parameters, TSFOutput * output) {
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


    // Transfer from device to host
    TubeSegmentation TS;
    TS.Fx = new float[totalSize];
    TS.Fy = new float[totalSize];
    TS.Fz = new float[totalSize];
    if(no3Dwrite || getParamBool(parameters, "32bit-vectors")) {
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
    }
    TS.radius = new float[totalSize];
    TS.TDF = new float[totalSize];
    ocl->queue.enqueueReadImage(*TDF, CL_TRUE, offset, region, 0, 0, TS.TDF);
    output->setTDF(TS.TDF);
    ocl->queue.enqueueReadImage(radius, CL_TRUE, offset, region, 0, 0, TS.radius);

    // Create pairs of voxels with high TDF
    std::vector<CrossSection *> crossSections = createGraph(TS, *size);

    // Display pairs
    SIPL::Volume<bool> * pairs = new SIPL::Volume<bool>(*size);
    pairs->fill(false);
    for(CrossSection * c : crossSections) {
    	pairs->set(c->pos, true);
    }
    pairs->showMIP();

    // Create segments from pairs
    std::vector<Segment *> segments = createSegments(TS, crossSections, *size);

    // Display segments

    /*
    Image3D * centerline = new Image3D;
    *centerline = runNewCenterlineAlg(*ocl, *size, parameters, vectorField, *TDF, radius, dataset);
    output->setCenterlineVoxels(centerline);

    Image3D * segmentation = new Image3D;
    if(!getParamBool(parameters, "no-segmentation")) {
    	if(!getParamBool(parameters, "sphere-segmentation")) {
			*segmentation = runInverseGradientSegmentation(*ocl, *centerline, vectorField, *size, parameters);
    	} else {
			*segmentation = runSphereSegmentation(*ocl, *centerline, radius, *size, parameters);
    	}
    	output->setSegmentation(segmentation);
    }

	if(getParamStr(parameters, "storage-dir") != "off") {
		writeDataToDisk(output, getParamStr(parameters, "storage-dir"));
    }
    */

}


void runCircleFittingAndRidgeTraversal(OpenCL * ocl, Image3D &dataset, SIPL::int3 * size, paramList &parameters, TSFOutput * output) {
    
    INIT_TIMER
    cl::Event startEvent, endEvent;
    cl_ulong start, end;
    Image3D vectorField, radius;
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
    if(no3Dwrite || getParamBool(parameters, "32bit-vectors")) {
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
    }
    TS.radius = new float[totalSize];
    TS.TDF = new float[totalSize];
    ocl->queue.enqueueReadImage(*TDF, CL_TRUE, offset, region, 0, 0, TS.TDF);
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
			*volume = runInverseGradientSegmentation(*ocl, *volume, vectorField, *size, parameters);
    	} else {
			*volume = runSphereSegmentation(*ocl,*volume, radius, *size, parameters);
    	}
		output->setSegmentation(volume);
    }


    if(getParamStr(parameters, "storage-dir") != "off") {
        writeDataToDisk(output, getParamStr(parameters, "storage-dir"));
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
        	char * str;
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

TSFOutput::TSFOutput(OpenCL * ocl, SIPL::int3 * size) {
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
		TDF = new float[size->x*size->y*size->z];
		ocl->queue.enqueueReadImage(*oclTDF,CL_TRUE, origin, region, 0, 0, TDF);		hostHasTDF = true;
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

