cmake_minimum_required(VERSION 2.8)
project(Tube-Segmentation-Framework)

add_subdirectory(SIPL)
include_directories(${SIPL_INCLUDE_DIRS})
add_subdirectory(OpenCLUtilities)
include_directories(${OPENCL_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D CPP11 -std=c++0x") # uncomment this to use boost libraries instead of C++11

find_package(OpenMP)
if(OPENMP_FOUND)
    message("-- OpenMP was detected. Using OpenMP to speed up some calculations.")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}" )
endif()

find_package(Boost COMPONENTS iostreams REQUIRED)

add_executable(tubeSegmentation main.cpp tube-segmentation.cpp histogram-pyramids.cpp parameters.cpp)
target_link_libraries(tubeSegmentation OCL-Utilities SIPL ${Boost_LIBRARIES})
