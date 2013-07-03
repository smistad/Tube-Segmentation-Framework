#ifndef RIDGE_TRAVERSAL_HPP
#define RIDGE_TRAVERSAL_HPP

#include "parameters.hpp"
#include "tube-segmentation.hpp"
#include "SIPL/Types.hpp"
#include <stack>

typedef struct CenterlinePoint {
    SIPL::int3 pos;
    bool large;
    CenterlinePoint * next;
} CenterlinePoint;

char * runRidgeTraversal(TubeSegmentation &T, SIPL::int3 size, paramList &parameters, std::stack<CenterlinePoint> centerlineStack);

#endif
