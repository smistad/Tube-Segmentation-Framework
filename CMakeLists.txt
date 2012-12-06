cmake_minimum_required(VERSION 2.8)
project(Tube-Segmentation-Framework)

#
# SIPL
###########
add_subdirectory(SIPL)
include_directories(${SIPL_INCLUDE_DIRS})

#
# OpenCLUtilities
###########
add_subdirectory(OpenCLUtilities)
include_directories(${OPENCL_INCLUDE_DIRS})

#
# Set compiler flags
###########
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -g -O3")

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

#
# Make configure file for find_package module 
###########
set( TSF_INCLUDE_DIRS
    ${Tube-Segmentation-Framework_SOURCE_DIR}
    ) 

set(TSF_LIBRARY_DIRS
    ${Tube-Segmentation-Framework_BINARY_DIR}
    )    

configure_file (
    "${PROJECT_SOURCE_DIR}/CMake/TSFConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/TSFConfig.cmake"
    )

#
# Include and linking directories
###########
include_directories(${Boost_INCLUDE_DIRS})
link_directories (${GTK2_LIBRARY_DIRS} ${Boost_LIBRARY_DIRS})

#
# Create executable
###########
add_executable(tubeSegmentation main.cpp tube-segmentation.cpp histogram-pyramids.cpp)
target_link_libraries(tubeSegmentation OCL-Utilities SIPL ${Boost_LIBRARIES})

#
# Create library
###########
add_library(tubeSegmentationLib tube-segmentation.cpp histogram-pyramids.cpp)
target_link_libraries(tubeSegmentationLib OCL-Utilities SIPL ${Boost_LIBRARIES})