#include "tests.hpp"
#include "../SIPL/Exceptions.hpp"

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

TEST(ParameterTest, NumericParameterValidation) {
	NumericParameter p;
	ASSERT_NO_THROW(p = NumericParameter(0.2, 0.1, 1.0, 0.1, "asd"));

	EXPECT_THROW(p.set(0.35), SIPL::SIPLException);
	EXPECT_THROW(p.set(0.05), SIPL::SIPLException);
	EXPECT_THROW(p.set(1.2), SIPL::SIPLException);
	EXPECT_NO_THROW(p.set(0.4));
}
