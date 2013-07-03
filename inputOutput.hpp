#ifndef INPUT_OUTPUT_H
#define INPUT_OUTPUT_H

#include "SIPL/Types.hpp"
#include <vector>
#include "parameters.hpp"
using namespace SIPL;

void writeToVtkFile(paramList &parameters, std::vector<int3> vertices, std::vector<SIPL::int2> edges);
#endif
