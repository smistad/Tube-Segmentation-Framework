#ifndef EIGEN_HESSIAN_H
#define EIGEN_HESSIAN_H
#include "SIPL/Types.hpp"
#include "tube-segmentation.hpp"
using namespace SIPL;

void doEigen(TubeSegmentation &T, int3 pos, int3 size, float3 * lambda, float3 * e1, float3 * e2, float3 * e3);

float3 getTubeDirection(TubeSegmentation &T, int3 pos, int3 size);

#endif
