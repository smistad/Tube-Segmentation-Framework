###########################################################
##          Tube-Segmentation-Framework project
###########################################################
project(Tube-Segmentation-Framework)

cmake_minimum_required(VERSION 2.8)

#
# Set compiler flags
###########
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D CP11 -std=c++0x") # uncomment this to use boost libraries instead of C++11

#------------------------------------------------------------------------------
# External libraries
#------------------------------------------------------------------------------

#
# OpenCLUtilities
###########
add_subdirectory(OpenCLUtilities)
#include_directories(${OPENCL_INCLUDE_DIRS})
find_package(OCL-Utilities PATHS "${Tube-Segmentation-Framework_BINARY_DIR}/OpenCLUtilities" REQUIRED)
include(${OCL-Utilities_USE_FILE})

#
# SIPL
###########
add_subdirectory(SIPL)
#include_directories(${SIPL_INCLUDE_DIRS})
find_package(SIPL PATHS "${Tube-Segmentation-Framework_BINARY_DIR}/SIPL" REQUIRED)
include(${SIPL_USE_FILE})

#
# OpenMP
###########
find_package(OpenMP)
if(OPENMP_FOUND)
    message("-- OpenMP was detected. Using OpenMP to speed up some calculations.")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}" )
endif()

#
# Boost
###########
find_package(Boost COMPONENTS iostreams REQUIRED)


#------------------------------------------------------------------------------
# Where to look for includes and libraries
#------------------------------------------------------------------------------
include_directories(${Boost_INCLUDE_DIRS})# ${OPENCL_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIRS}) #${GTK2_LIBRARY_DIRS})


#------------------------------------------------------------------------------
# Targets
#------------------------------------------------------------------------------

#
# tubeSegmentationLib library
###########
add_library(tubeSegmentationLib tube-segmentation.cpp histogram-pyramids.cpp)
target_link_libraries(tubeSegmentationLib OCL-Utilities SIPL ${Boost_LIBRARIES} ${OPENCL_LIBRARIES})

#
# tubeSegmentation executable
###########
add_executable(tubeSegmentation main.cpp tube-segmentation.cpp histogram-pyramids.cpp)
target_link_libraries(tubeSegmentation SIPL OCL-Utilities ${Boost_LIBRARIES} ${OPENCL_LIBRARIES})
