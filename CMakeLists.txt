cmake_minimum_required(VERSION 2.8)
project(Tube-Segmentation-Framework)

add_subdirectory(SIPL)
include_directories(${SIPL_INCLUDE_DIRS})
add_subdirectory(OpenCLUtilities)
include_directories(${OPENCL_INCLUDE_DIRS})

add_executable(tubeSegmentation main.cpp tube-segmentation.cpp)
target_link_libraries(tubeSegmentation OCL-Utilities SIPL )
