#include "tests.hpp"
#include "../tube-segmentation.cpp"
#include "../parameters.hpp"
#include "../SIPL/Exceptions.hpp"
#include "tubeValidation.cpp"


TEST(TubeSegmentation, WrongFilenameException) {
	paramList parameters = initParameters();
	ASSERT_THROW(run("somefilethatdoesntexist.mhd", parameters), SIPL::IOException);
}

#define TESTDATA_PATH "/home/smistad/Dropbox/TestData/"
TEST(TubeSegmentation, SystemTestWithSyntheticData) {
	paramList parameters = initParameters();
	parameters = setParameter(parameters, "parameters", "vascusynth");
	parameters = loadParameterPreset(parameters);
	TSFOutput * output;
	(output = run(std::string(TESTDATA_PATH) + std::string("synthetic/noisy.mhd"), parameters));

	TubeValidation result = validateTube(
			output,
			std::string(TESTDATA_PATH) + std::string("synthetic/original.mhd"),
			std::string(TESTDATA_PATH) + std::string("synthetic/real_centerline.mhd")
	);

	EXPECT_GT(1.5, result.averageDistanceFromCenterline);
	EXPECT_LT(60.0, result.percentageExtractedCenterlines);
	EXPECT_LT(0.7, result.precision);
	EXPECT_LT(0.7, result.recall);
}
