###########################################################
##          Tube-Segmentation-Framework project
###########################################################
project(Tube-Segmentation-Framework)

cmake_minimum_required(VERSION 2.8)

#
# Set compiler flags
###########
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3")
option(USE_C++11 "use C++11 (Tube-Segmentation-Framework)" ON) # cmake option to use boost libraries instead of C++11
if(USE_C++11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D CPP11 -std=c++0x")
endif()

#------------------------------------------------------------------------------
# External libraries
#------------------------------------------------------------------------------

#
# OpenCLUtilities
###########
add_subdirectory(OpenCLUtilities)
find_package(OCL-Utilities PATHS "${Tube-Segmentation-Framework_BINARY_DIR}/OpenCLUtilities" REQUIRED)
include(${OCL-Utilities_USE_FILE})

#
# SIPL
###########
add_subdirectory(SIPL)
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
include_directories(${Boost_INCLUDE_DIRS} ${PROJECT_BINARY_DIR})
link_directories(${Boost_LIBRARY_DIRS})


#------------------------------------------------------------------------------
# Targets
#------------------------------------------------------------------------------

#
# tubeSegmentationLib library
###########
add_library(tubeSegmentationLib tube-segmentation.cpp histogram-pyramids.cpp parameters.cpp gradientVectorFlow.cpp commons.cpp tubeDetectionFilters.cpp ridgeTraversalCenterlineExtraction.cpp eigenanalysisOfHessian.cpp globalCenterlineExtraction.cpp)
target_link_libraries(tubeSegmentationLib OCL-Utilities SIPL ${Boost_LIBRARIES} ${OPENCL_LIBRARIES})

#
# tubeSegmentation executable
###########
if(SIPL_USE_GTK)
    add_executable(tubeSegmentation main.cpp tube-segmentation.cpp histogram-pyramids.cpp parameters.cpp gradientVectorFlow.cpp commons.cpp tubeDetectionFilters.cpp ridgeTraversalCenterlineExtraction.cpp eigenanalysisOfHessian.cpp globalCenterlineExtraction.cpp)
    target_link_libraries(tubeSegmentation SIPL OCL-Utilities ${Boost_LIBRARIES} ${OPENCL_LIBRARIES})
endif()

#------------------------------------------------------------------------------
# Testing
#------------------------------------------------------------------------------
find_package(GTest)
if(GTEST_FOUND)
if(SIPL_USE_GTK)
	message("Google test framework found. Enabling testing ...")
	target_link_libraries(tubeSegmentationLib OCL-Utilities SIPL ${Boost_LIBRARIES})
	enable_testing()
	add_subdirectory(tests)
endif()
endif()

#------------------------------------------------------------------------------
# Set variables
#------------------------------------------------------------------------------

set(PARAMETERS_DIR ${PROJECT_SOURCE_DIR}/parameters)
set(KERNELS_DIR ${PROJECT_SOURCE_DIR})
set(TESTDATA_DIR ${PROJECT_SOURCE_DIR}/tests/data)

#------------------------------------------------------------------------------
# Configure file for find_package module 
#------------------------------------------------------------------------------

set( Tube-Segmentation-Framework_INCLUDE_DIRS
    ${Tube-Segmentation-Framework_SOURCE_DIR}
) 

set(Tube-Segmentation-Framework_LIBRARY_DIRS
    ${Tube-Segmentation-Framework_BINARY_DIR}
    ${OCL-Utilities_LIBRARY_DIRS}
    ${SIPL_BINARY_LIBRARY_DIRS}
)    

configure_file (
    "${PROJECT_SOURCE_DIR}/CMake/Tube-Segmentation-FrameworkConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/Tube-Segmentation-FrameworkConfig.cmake"
)
 
#------------------------------------------------------------------------------
# Configure file for settings, filepaths, parameters ...
#------------------------------------------------------------------------------

configure_file(
 "${PROJECT_SOURCE_DIR}/CMake/tsf-config.h.in"
 "${PROJECT_BINARY_DIR}/tsf-config.h"
)
