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

TubeValidation runSyntheticData(paramList parameters) {
	TSFOutput * output;
	(output = run(std::string(TESTDATA_PATH) + std::string("synthetic/noisy.mhd"), parameters));

	TubeValidation result = validateTube(
			output,
			std::string(TESTDATA_PATH) + std::string("synthetic/original.mhd"),
			std::string(TESTDATA_PATH) + std::string("synthetic/real_centerline.mhd")
	);

	delete output;
	return result;
}

TEST(TubeSegmentation, SystemTestWithSyntheticDataPCE) {
	paramList parameters = initParameters();
	parameters = setParameter(parameters, "parameters", "vascusynth");
	parameters = loadParameterPreset(parameters);
	TubeValidation result;

	// Normal execution
	result = runSyntheticData(parameters);
	EXPECT_GT(1.5, result.averageDistanceFromCenterline);
	EXPECT_LT(79.0, result.percentageExtractedCenterlines);
	EXPECT_LT(0.7, result.precision);
	EXPECT_LT(0.7, result.recall);

	// 32 bit 3D textures
	parameters = setParameter(parameters, "buffers-only", "false");
	parameters = setParameter(parameters, "32bit-vectors", "true");
	result = runSyntheticData(parameters);
	EXPECT_GT(1.5, result.averageDistanceFromCenterline);
	EXPECT_LT(79.0, result.percentageExtractedCenterlines);
	EXPECT_LT(0.7, result.precision);
	EXPECT_LT(0.7, result.recall);

	// 32 bit buffers
	parameters = setParameter(parameters, "buffers-only", "true");
	parameters = setParameter(parameters, "32bit-vectors", "true");
	result = runSyntheticData(parameters);
	EXPECT_GT(1.5, result.averageDistanceFromCenterline);
	EXPECT_LT(79.0, result.percentageExtractedCenterlines);
	EXPECT_LT(0.7, result.precision);
	EXPECT_LT(0.7, result.recall);

	// 16 bit 3D textures
	parameters = setParameter(parameters, "buffers-only", "false");
	parameters = setParameter(parameters, "32bit-vectors", "false");
	result = runSyntheticData(parameters);
	EXPECT_GT(1.5, result.averageDistanceFromCenterline);
	EXPECT_LT(79.0, result.percentageExtractedCenterlines);
	EXPECT_LT(0.7, result.precision);
	EXPECT_LT(0.7, result.recall);

	// 16 bit buffers
	parameters = setParameter(parameters, "buffers-only", "true");
	parameters = setParameter(parameters, "32bit-vectors", "false");
	result = runSyntheticData(parameters);
	EXPECT_GT(1.5, result.averageDistanceFromCenterline);
	EXPECT_LT(79.0, result.percentageExtractedCenterlines);
	EXPECT_LT(0.7, result.precision);
	EXPECT_LT(0.7, result.recall);
}

TEST(TubeSegmentation, SystemTestWithSyntheticDataRidgeTraversal) {
	paramList parameters = initParameters();
	parameters = setParameter(parameters, "parameters", "vascusynth");
	parameters = setParameter(parameters, "centerline-method", "ridge");
	parameters = loadParameterPreset(parameters);

	TubeValidation result;

	result = runSyntheticData(parameters);
	EXPECT_GT(1.5, result.averageDistanceFromCenterline);
	EXPECT_LT(75.0, result.percentageExtractedCenterlines);
	EXPECT_LT(0.7, result.precision);
	EXPECT_LT(0.6, result.recall);

	// 32 bit 3D textures
	parameters = setParameter(parameters, "buffers-only", "false");
	parameters = setParameter(parameters, "32bit-vectors", "true");
	result = runSyntheticData(parameters);
	EXPECT_GT(1.5, result.averageDistanceFromCenterline);
	EXPECT_LT(75.0, result.percentageExtractedCenterlines);
	EXPECT_LT(0.7, result.precision);
	EXPECT_LT(0.6, result.recall);

	// 32 bit buffers
	parameters = setParameter(parameters, "buffers-only", "true");
	parameters = setParameter(parameters, "32bit-vectors", "true");
	result = runSyntheticData(parameters);
	EXPECT_GT(1.5, result.averageDistanceFromCenterline);
	EXPECT_LT(75.0, result.percentageExtractedCenterlines);
	EXPECT_LT(0.7, result.precision);
	EXPECT_LT(0.6, result.recall);

	// 16 bit 3D textures
	parameters = setParameter(parameters, "buffers-only", "false");
	parameters = setParameter(parameters, "32bit-vectors", "false");
	result = runSyntheticData(parameters);
	EXPECT_GT(1.5, result.averageDistanceFromCenterline);
	EXPECT_LT(75.0, result.percentageExtractedCenterlines);
	EXPECT_LT(0.7, result.precision);
	EXPECT_LT(0.6, result.recall);

	// 16 bit buffers
	parameters = setParameter(parameters, "buffers-only", "true");
	parameters = setParameter(parameters, "32bit-vectors", "false");
	result = runSyntheticData(parameters);
	EXPECT_GT(1.5, result.averageDistanceFromCenterline);
	EXPECT_LT(75.0, result.percentageExtractedCenterlines);
	EXPECT_LT(0.7, result.precision);
	EXPECT_LT(0.6, result.recall);
}
