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

TEST(TSFOutputTest, GetHostData) {
	TSFOutput output(new OpenCL, new SIPL::int3);
	float * TDF = new float[3];
	TDF[0] = 0.5f;
	TDF[1] = 1.0f;
	TDF[2] = 0.0f;
	output.setTDF(TDF);
	ASSERT_EQ(output.getTDF()[0], 0.5f);
	ASSERT_EQ(output.getTDF()[1], 1.0f);
	ASSERT_EQ(output.getTDF()[2], 0.0f);

	char * data = new char[3];
	data[0] = 1;
	data[1] = 2;
	data[2] = 100;
	output.setSegmentation(data);
	ASSERT_EQ(output.getSegmentation()[0], 1);
	ASSERT_EQ(output.getSegmentation()[1], 2);
	ASSERT_EQ(output.getSegmentation()[2], 100);

	char * data2 = new char[3];
	data2[0] = 10;
	data2[1] = 20;
	data2[2] = 10;
	output.setCenterlineVoxels(data2);
	ASSERT_EQ(output.getCenterlineVoxels()[0], 10);
	ASSERT_EQ(output.getCenterlineVoxels()[1], 20);
	ASSERT_EQ(output.getCenterlineVoxels()[2], 10);
}

TEST(TSFOutputTest, GetSize) {
	SIPL::int3 * size = new SIPL::int3(100, 20, 1);
	TSFOutput output(new OpenCL, size);
	ASSERT_EQ(output.getSize()->x, 100);
	ASSERT_EQ(output.getSize()->y, 20);
	ASSERT_EQ(output.getSize()->z, 1);
}

