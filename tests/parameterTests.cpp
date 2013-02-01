#include "tests.hpp"

// Tests for the parameter system

TEST(ParameterTest, GetDefaultParameters) {
	paramList parameters = initParameters();

	EXPECT_FALSE(getParamBool(parameters, "display"));
	EXPECT_EQ("gpu", getParamStr(parameters, "device"));
	EXPECT_EQ(0.05f, getParam(parameters, "gvf-mu"));
}

TEST(ParameterTest, SetParameters) {
	paramList parameters = initParameters();

	parameters = setParameter(parameters, "display", "true");
	EXPECT_TRUE(getParamBool(parameters, "display"));
	parameters = setParameter(parameters, "cropping", "lung");
	EXPECT_EQ("lung", getParamStr(parameters, "cropping"));
	parameters = setParameter(parameters, "tdf-high", "0.9");
	EXPECT_EQ(0.9f, getParam(parameters, "tdf-high"));
}
