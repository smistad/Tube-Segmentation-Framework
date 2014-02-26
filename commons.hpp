#ifndef COMMONS_H
#define COMMONS_H
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include "OpenCLManager.hpp"
#include "Context.hpp"
#include "SIPL/Types.hpp"

// TODO The use of this struct will be removed eventually
typedef struct OpenCL {
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Device device;
    cl::Platform platform;
    oul::GarbageCollector * GC;
    oul::Context oulContext;
} OpenCL;

#ifdef WIN32
// Add some math functions that are missing from the windows math library
template <class T>
static inline double log2(T a) {
	return log((double)a)/log(2.0);
}

template <class T>
static inline double round(T a) {
	return floor((double)a+0.5);
}
#endif

static inline bool inBounds(SIPL::int3 pos, SIPL::int3 size) {
    return pos.x > 0 && pos.y > 0 && pos.z > 0 && pos.x < size.x && pos.y < size.y && pos.z < size.z;
}
#endif
