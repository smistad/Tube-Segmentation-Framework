#include "tests.hpp"

TEST(TSFOutputTest, Initialization) {
	TSFOutput output(oul::DeviceCriteria(), new SIPL::int3);
	EXPECT_FALSE(output.hasSegmentation());
	EXPECT_FALSE(output.hasCenterlineVoxels());
	EXPECT_FALSE(output.hasTDF());
}

TEST(TSFOutputTest, SetHostData) {
	TSFOutput output(oul::DeviceCriteria(), new SIPL::int3);
	output.setTDF(new float);
	EXPECT_TRUE(output.hasTDF());
	EXPECT_FALSE(output.hasSegmentation());
	EXPECT_FALSE(output.hasCenterlineVoxels());
	output.setSegmentation(new char);
	EXPECT_TRUE(output.hasTDF());
	EXPECT_TRUE(output.hasSegmentation());
	EXPECT_FALSE(output.hasCenterlineVoxels());
	output.setCenterlineVoxels(new char);
	EXPECT_TRUE(output.hasTDF());
	EXPECT_TRUE(output.hasSegmentation());
	EXPECT_TRUE(output.hasCenterlineVoxels());
}

TEST(TSFOutputTest, GetHostData) {
	TSFOutput output(oul::DeviceCriteria(), new SIPL::int3);
	float * TDF = new float[3];
	TDF[0] = 0.5f;
	TDF[1] = 1.0f;
	TDF[2] = 0.0f;
	output.setTDF(TDF);
	EXPECT_EQ(0.5f, output.getTDF()[0]);
	EXPECT_EQ(1.0f, output.getTDF()[1]);
	EXPECT_EQ(0.0f, output.getTDF()[2]);

	char * data = new char[3];
	data[0] = 1;
	data[1] = 2;
	data[2] = 100;
	output.setSegmentation(data);
	EXPECT_EQ(1, output.getSegmentation()[0]);
	EXPECT_EQ(2, output.getSegmentation()[1]);
	EXPECT_EQ(100, output.getSegmentation()[2]);

	char * data2 = new char[3];
	data2[0] = 10;
	data2[1] = 20;
	data2[2] = 10;
	output.setCenterlineVoxels(data2);
	EXPECT_EQ(10, output.getCenterlineVoxels()[0]);
	EXPECT_EQ(20, output.getCenterlineVoxels()[1]);
	EXPECT_EQ(10, output.getCenterlineVoxels()[2]);
}

TEST(TSFOutputTest, GetSize) {
	SIPL::int3 * size = new SIPL::int3(100, 20, 1);
	TSFOutput output(oul::DeviceCriteria(), size);
	EXPECT_EQ(100, output.getSize()->x);
	EXPECT_EQ(20, output.getSize()->y);
	EXPECT_EQ(1, output.getSize()->z);
}

TEST(TSFOutputTest, ShiftVector) {
	SIPL::int3 shiftVector(3, 10, 2);
	TSFOutput output(oul::DeviceCriteria(), new SIPL::int3);
	output.setShiftVector(shiftVector);
	EXPECT_EQ(3, output.getShiftVector().x);
	EXPECT_EQ(10, output.getShiftVector().y);
	EXPECT_EQ(2, output.getShiftVector().z);
}

