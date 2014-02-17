#include "../SIPL/Core.hpp"
#include "../tube-segmentation.hpp"
#include <iostream>
#include <queue>
#include <vector>
#include <string>
using namespace SIPL;

typedef struct TubeValidation {
	float averageDistanceFromCenterline;
	float percentageExtractedCenterlines;
	float recall;
	float precision;
	unsigned int incorrectCenterpoints;
} TubeValidation;

TubeValidation getValidationMeasures(
		Volume<char> * realCenterlines,
		Volume<char> * eCenterlines,
		Volume<char> * original,
		Volume<char> * segmentation,
		bool visualize
		) {
	TubeValidation result;
	Volume<char> * detectedCenterlines = new Volume<char>(realCenterlines->getSize());
	detectedCenterlines->fill(0);

	float avgDistance = 0.0f;
	int counter = 0;
	unsigned int incorrectCenterpoints = 0;
	int width = eCenterlines->getWidth();
	int height = eCenterlines->getHeight();
	int depth = eCenterlines->getDepth();
	int maxRadius = 4;
	for(int z = 0; z < depth; z++) {
	for(int y = 0; y < height; y++) {
	for(int x = 0; x < width; x++) {
		if(eCenterlines->get(x,y,z) == 1) {
			int3 start(x,y,z);
			bool found = false;
			float bestDistance = 9999.0f;

			for(int a = -maxRadius; a <= maxRadius; a++) {
			for(int b = -maxRadius; b <= maxRadius; b++) {
			for(int c = -maxRadius; c <= maxRadius; c++) {
				int3 n = start + int3(a,b,c);
				if(eCenterlines->inBounds(n) &&
						start.distance(n) < maxRadius) {
					if(realCenterlines->get(n) == 1) {
						found = true;
						if(start.distance(n) < bestDistance) {
							bestDistance = start.distance(n);
						}
						detectedCenterlines->set(n, 1);
					}
				}
			}}}
			if(!found) {
				incorrectCenterpoints++;
			} else {
				avgDistance += bestDistance;
				counter++;
			}
		}
	}}}
	avgDistance /= counter;
	result.averageDistanceFromCenterline = avgDistance;
	result.incorrectCenterpoints = incorrectCenterpoints;


	std::cout << std::endl;
	std::cout << "Centerline result" << std::endl << "--------------------" << std::endl;
	std::cout << "Average distance from real centerline was: " << avgDistance << " voxels" << std::endl;
	std::cout << "Number of incorrect centerpoints: " << incorrectCenterpoints << std::endl;

	int detected = 0;
	int undetected = 0;
	Volume<float3> * visualization = new Volume<float3>(detectedCenterlines->getSize());
	for(int i = 0; i < visualization->getTotalSize(); i++) {
		float3 v;
		if(detectedCenterlines->get(i) == 1) { // flag detected point
			v.x = 1.0f;
			detected++;
		}
		if(realCenterlines->get(i) == 1 && detectedCenterlines->get(i) == 0) { // undetected centerpoint
			v.z = 1.0f;
			undetected++;
		}
		//if(eCenterlines->get(i) == 1) // flag the actual extracted points
		//	v.y = 1.0f;
		visualization->set(i, v);
	}
	if(visualize)
		visualization->displayMIP();

	result.percentageExtractedCenterlines = (float)detected*100.0f/(detected+undetected);
	std::cout << "Percentage of extracted centerline: " << (float)detected*100.0f/(detected+undetected) << std::endl << std::endl;

	int extracted = 0;
	for(int i = 0; i < eCenterlines->getTotalSize(); i++) {
		if(eCenterlines->get(i) == 1)
			extracted++;
	}
	std::cout << "Total nr of real centerpoints: " << detected+undetected << std::endl;
	std::cout << "Total nr of extracted centerpoints: " << extracted << std::endl;
	std::cout << "Ratio: " << (float)extracted/(detected+undetected) << std::endl << std::endl;



	// for segmentation measure: true positives, false positives, false negatives, false positives
	int truePositives = 0;
	int falsePositives = 0;
	int trueNegatives = 0;
	int falseNegatives = 0;
	for(int i = 0; i < original->getTotalSize(); i++) {
		bool truth = original->get(i) == 1;
		bool test = segmentation->get(i) == 1;
		if(truth && test) {
			truePositives++;
		} else if(!truth && test) {
			falsePositives++;
		} else if(!truth && !test) {
			trueNegatives++;
		} else {
			falseNegatives++;
		}
	}
	std::cout << "Segmentation result" << std::endl << "--------------------" << std::endl;
	std::cout << "True positives: " << truePositives << std::endl;
	std::cout << "False positives: " << falsePositives << std::endl;
	std::cout << "True negatives: " << trueNegatives << std::endl;
	std::cout << "False negatives: " << falseNegatives << std::endl << std::endl;

	result.recall = (float)truePositives / (truePositives+falseNegatives);
	result.precision = (float)truePositives / (truePositives+falsePositives);
	std::cout << "Recall: " << result.recall << std::endl;
	std::cout << "Precision: " << result.precision << std::endl;
	return result;
}


TubeValidation validateTube(TSFOutput * output, std::string segmentationPath, std::string centerlinePath) {

	Volume<char> * realCenterlines = new Volume<char>(centerlinePath.c_str());

	Volume<char> * original = new Volume<char>(segmentationPath.c_str());


	// load extracted centerlines
	Volume<char> * extractedCenterlines = new Volume<char>(*(output->getSize()));
	extractedCenterlines->setData(output->getCenterlineVoxels());

	Volume<char> * segmentation = new Volume<char>(*(output->getSize()));
	segmentation->setData(output->getSegmentation());

	TubeValidation result = getValidationMeasures(realCenterlines, extractedCenterlines, original, segmentation, false);

	return result;

}
