#include "tests.hpp"

TEST(TSFOutputTest, Initialization) {
	TSFOutput output(new OpenCL, new SIPL::int3);
	ASSERT_EQ(output.hasSegmentation(), false);
	ASSERT_EQ(output.hasCenterlineVoxels(), false);
	ASSERT_EQ(output.hasTDF(), false);
}

TEST(TSFOutputTest, SetHostData) {
	TSFOutput output(new OpenCL, new SIPL::int3);
	output.setTDF(new float);
	ASSERT_EQ(output.hasTDF(), true);
	ASSERT_EQ(output.hasSegmentation(), false);
	ASSERT_EQ(output.hasCenterlineVoxels(), false);
	output.setSegmentation(new char);
	ASSERT_EQ(output.hasTDF(), true);
	ASSERT_EQ(output.hasSegmentation(), true);
	ASSERT_EQ(output.hasCenterlineVoxels(), false);
	output.setCenterlineVoxels(new char);
	ASSERT_EQ(output.hasTDF(), true);
	ASSERT_EQ(output.hasSegmentation(), true);
	ASSERT_EQ(output.hasCenterlineVoxels(), true);
}

TEST(TSFOutputTest, GetSize) {
	SIPL::int3 * size = new SIPL::int3(100, 20, 1);
	TSFOutput output(new OpenCL, size);
	ASSERT_EQ(output.getSize()->x, 100);
	ASSERT_EQ(output.getSize()->y, 20);
	ASSERT_EQ(output.getSize()->z, 1);
}

