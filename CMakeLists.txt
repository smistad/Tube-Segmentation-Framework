###########################################################
##          Tube-Segmentation-Framework project
###########################################################
project(Tube-Segmentation-Framework)

cmake_minimum_required(VERSION 2.8)

#------------------------------------------------------------------------------
# Set compiler flags
#------------------------------------------------------------------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3")
option(USE_C++11 "use C++11 (Tube-Segmentation-Framework)" ON) # cmake option to use boost libraries instead of C++11
if(USE_C++11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D CPP11 -std=c++0x")
endif()


#------------------------------------------------------------------------------
# External libraries
#------------------------------------------------------------------------------
option (TSF_USE_EXTRNAL_OUL "Use external OpenCLUtilityLibrary" OFF)
option (TSF_EXTERNAL_OUL_PATH "Path to external OpenCLUtilityLibrary build" "")
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/CMake
    )
include (tsfInitializeLibraries)
tsf_initialize_external_libraries()


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
add_library(tubeSegmentationLib 
	tube-segmentation.cpp 
	parameters.cpp 
	gradientVectorFlow.cpp 
	tubeDetectionFilters.cpp 
	ridgeTraversalCenterlineExtraction.cpp 
	eigenanalysisOfHessian.cpp
	globalCenterlineExtraction.cpp 
	parallelCenterlineExtraction.cpp 
	inputOutput.cpp
	segmentation.cpp
)
target_link_libraries(tubeSegmentationLib OpenCLUtilityLibrary SIPL ${Boost_LIBRARIES} ${OPENCL_LIBRARIES})

#
# tubeSegmentation executable
###########
option(SIPL_USE_GTK "Use SIPL with GTK" ON)
if(SIPL_USE_GTK)
    add_executable(tubeSegmentation
    	main.cpp
		tube-segmentation.cpp 
		parameters.cpp 
		gradientVectorFlow.cpp 
		tubeDetectionFilters.cpp 
		ridgeTraversalCenterlineExtraction.cpp 
		eigenanalysisOfHessian.cpp
		globalCenterlineExtraction.cpp 
		parallelCenterlineExtraction.cpp 
		inputOutput.cpp
		segmentation.cpp
	)
    target_link_libraries(tubeSegmentation SIPL OpenCLUtilityLibrary ${Boost_LIBRARIES} ${OPENCL_LIBRARIES})
endif()

#------------------------------------------------------------------------------
# Testing
#------------------------------------------------------------------------------

tsf_enable_testing()


#------------------------------------------------------------------------------
# Set variables
#------------------------------------------------------------------------------

set(PARAMETERS_DIR ${PROJECT_SOURCE_DIR}/parameters)
set(KERNELS_DIR ${PROJECT_SOURCE_DIR})
set(TESTDATA_DIR ${PROJECT_SOURCE_DIR}/tests/data)
if(TSF_USE_EXTRNAL_OUL)
set(OUL_DIR ${TSF_EXTERNAL_OUL_PATH})
else(TSF_USE_EXTRNAL_OUL)
set(OUL_DIR ${PROJECT_SOURCE_DIR}/OpenCLUtilityLibrary)
endif(TSF_USE_EXTRNAL_OUL)

#------------------------------------------------------------------------------
# Configure file for find_package module 
#------------------------------------------------------------------------------

set( Tube-Segmentation-Framework_MODULE_PATH
    ${Tube-Segmentation-Framework_SOURCE_DIR}/CMake
)

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
