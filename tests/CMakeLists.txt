cmake_minimum_required(VERSION 2.8)

# Locate GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D CPP11 -std=c++0x")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
# Link runTests with what we want to test and the GTest and pthread library
add_executable(runTests tests.cpp)
target_link_libraries(runTests tubeSegmentationLib ${GTEST_LIBRARIES} pthread)

add_test(runTests runTests)