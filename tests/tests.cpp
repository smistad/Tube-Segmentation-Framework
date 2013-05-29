#include "tests.hpp"

// Include all the tests here
#include "TSFOutputTests.cpp"
#include "parameterTests.cpp"
#include "tubeSegmentationTests.cpp"
#include "clinicalTests.cpp"

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

