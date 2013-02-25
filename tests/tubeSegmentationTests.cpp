#include "tests.hpp"
#include "../tube-segmentation.cpp"
#include "../parameters.hpp"
#include "../SIPL/Exceptions.hpp"
#include "tubeValidation.cpp"
#include "../tsf-config.h"


TEST(TubeSegmentation, WrongFilenameException) {
	paramList parameters = initParameters(PARAMETERS_DIR);
	ASSERT_THROW(run("somefilethatdoesntexist.mhd", parameters, KERNELS_DIR), SIPL::IOException);
}


class TubeSegmentationPCE : public ::testing::Test {
protected:
	virtual void SetUp() {
		parameters = initParameters(PARAMETERS_DIR);
		parameters = setParameter(parameters, "parameters", "Synthetic-Vascusynth");
		parameters = setParameter(parameters, "centerline-method", "gpu");
		parameters = loadParameterPreset(parameters, PARAMETERS_DIR);
	};
	virtual void TearDown() {

	};
	paramList parameters;
	TubeValidation result;
};

class TubeSegmentationRidge : public ::testing::Test {
protected:
	virtual void SetUp() {
		parameters = initParameters(PARAMETERS_DIR);
		parameters = setParameter(parameters, "parameters", "Synthetic-Vascusynth");
		parameters = setParameter(parameters, "centerline-method", "ridge");
		parameters = loadParameterPreset(parameters, PARAMETERS_DIR);
	};
	virtual void TearDown() {

	};
	paramList parameters;
	TubeValidation result;
};


TubeValidation runSyntheticData(paramList parameters) {
	std::string datasetNr = "1";
	TSFOutput * output;
	output = run(std::string(TESTDATA_DIR) + std::string("/synthetic/dataset_") + datasetNr + std::string("/noisy.mhd"), parameters, KERNELS_DIR);

	TubeValidation result = validateTube(
			output,
			std::string(TESTDATA_DIR) + std::string("/synthetic/dataset_") + datasetNr + std::string("/original.mhd"),
			std::string(TESTDATA_DIR) + std::string("/synthetic/dataset_") + datasetNr + std::string("/real_centerline.mhd")
	);

	delete output;
	return result;
}


TEST_F(TubeSegmentationPCE, SystemTestWithSyntheticDataNormal) {
	// Normal execution
	result = runSyntheticData(parameters);
	parameters = setParameter(parameters, "buffers-only", "false");
	parameters = setParameter(parameters, "32bit-vectors", "false");
	EXPECT_GT(1.5, result.averageDistanceFromCenterline);
	EXPECT_LT(75.0, result.percentageExtractedCenterlines);
	EXPECT_LT(0.7, result.precision);
	EXPECT_LT(0.7, result.recall);
}

TEST_F(TubeSegmentationPCE, SystemTestWithSyntheticData32bit) {
	// 32 bit 3D textures
	parameters = setParameter(parameters, "buffers-only", "false");
	parameters = setParameter(parameters, "32bit-vectors", "true");
	result = runSyntheticData(parameters);
	EXPECT_GT(1.5, result.averageDistanceFromCenterline);
	EXPECT_LT(75.0, result.percentageExtractedCenterlines);
	EXPECT_LT(0.7, result.precision);
	EXPECT_LT(0.7, result.recall);
}

TEST_F(TubeSegmentationPCE, SystemTestWithSyntheticData32bitBuffers) {
	// 32 bit buffers
	parameters = setParameter(parameters, "buffers-only", "true");
	parameters = setParameter(parameters, "32bit-vectors", "true");
	result = runSyntheticData(parameters);
	EXPECT_GT(1.5, result.averageDistanceFromCenterline);
	EXPECT_LT(75.0, result.percentageExtractedCenterlines);
	EXPECT_LT(0.7, result.precision);
	EXPECT_LT(0.7, result.recall);
}

TEST_F(TubeSegmentationPCE, SystemTestWithSyntheticData16bitBuffers) {
	// 16 bit buffers
	parameters = setParameter(parameters, "buffers-only", "true");
	parameters = setParameter(parameters, "32bit-vectors", "false");
	result = runSyntheticData(parameters);
	EXPECT_GT(1.5, result.averageDistanceFromCenterline);
	EXPECT_LT(75.0, result.percentageExtractedCenterlines);
	EXPECT_LT(0.7, result.precision);
	EXPECT_LT(0.7, result.recall);
}

TEST_F(TubeSegmentationRidge, SystemTestWithSyntheticDataNormal) {
	// Normal execution
	result = runSyntheticData(parameters);
	parameters = setParameter(parameters, "buffers-only", "false");
	parameters = setParameter(parameters, "32bit-vectors", "false");
	EXPECT_GT(0.5, result.averageDistanceFromCenterline);
	EXPECT_LT(75.0, result.percentageExtractedCenterlines);
	EXPECT_LT(0.7, result.precision);
	EXPECT_LT(0.6, result.recall);
}

TEST_F(TubeSegmentationRidge, SystemTestWithSyntheticData32bit) {
	// 32 bit 3D textures
	parameters = setParameter(parameters, "buffers-only", "false");
	parameters = setParameter(parameters, "32bit-vectors", "true");
	result = runSyntheticData(parameters);
	EXPECT_GT(0.5, result.averageDistanceFromCenterline);
	EXPECT_LT(75.0, result.percentageExtractedCenterlines);
	EXPECT_LT(0.7, result.precision);
	EXPECT_LT(0.6, result.recall);
}

TEST_F(TubeSegmentationRidge, SystemTestWithSyntheticData32bitBuffers) {
	// 32 bit buffers
	parameters = setParameter(parameters, "buffers-only", "true");
	parameters = setParameter(parameters, "32bit-vectors", "true");
	result = runSyntheticData(parameters);
	EXPECT_GT(0.5, result.averageDistanceFromCenterline);
	EXPECT_LT(75.0, result.percentageExtractedCenterlines);
	EXPECT_LT(0.7, result.precision);
	EXPECT_LT(0.6, result.recall);
}

TEST_F(TubeSegmentationRidge, SystemTestWithSyntheticData16bitBuffers) {
	// 16 bit buffers
	parameters = setParameter(parameters, "buffers-only", "true");
	parameters = setParameter(parameters, "32bit-vectors", "false");
	result = runSyntheticData(parameters);
	EXPECT_GT(0.5, result.averageDistanceFromCenterline);
	EXPECT_LT(75.0, result.percentageExtractedCenterlines);
	EXPECT_LT(0.7, result.precision);
	EXPECT_LT(0.6, result.recall);
}

