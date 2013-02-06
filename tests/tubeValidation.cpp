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
} TubeValidation;

TubeValidation validateTube(TSFOutput * output, std::string segmentationPath, std::string centerlinePath) {
	TubeValidation result;

	// First copy over the generated datasets:
	// mv /home/smistad/Dropbox/Programmering/VascuSynth/centerline.raw /home/smistad/Dropbox/Programmering/Tube-Validation/real_centerline.raw; mv /home/smistad/Dropbox/Programmering/VascuSynth/original.raw /home/smistad/Dropbox/Programmering/Tube-Validation/original.raw; mv /home/smistad/Dropbox/Programmering/VascuSynth/noisy0.raw /home/smistad/Dropbox/Programmering/Tube-Validation/noisy0.raw
	// Secondy run the tube segmentation program, alter between --centerline-method ridge and --16bit-vectors
	// /home/smistad/Dropbox/Programmering/Tube-Segmentation-Framework/tubeSegmentation /home/smistad/Dropbox/Programmering/Tube-Validation/noisy0.mhd --parameters vascusynth --storage-dir /home/smistad/Dropbox/Programmering/Tube-Validation/
	// Then run this program.

	Volume<char> * realCenterlines = new Volume<char>(centerlinePath.c_str());
	Volume<char> * detectedCenterlines = new Volume<char>(realCenterlines);

	Volume<uchar> * original = new Volume<uchar>(segmentationPath.c_str());

	/*// For visualization
	Volume<float2> * combined = new Volume<float2>(original->getSize());
	for(int i = 0; i < original->getTotalSize();i++) {
		float2 v;
		v.x = 0.0;
		if(original->get(i) > 0)
			v.x = 1.0f;
		v.y = centerlines->get(i);
		combined->set(i, v);
	}
	combined->showMIP();
	*/

	// load extracted centerlines
	Volume<char> * extractedCenterlines = new Volume<char>(*(output->getSize()));
	extractedCenterlines->setData(output->getCenterlineVoxels());

	// For each extracted centerline point, find the closest real centerpoint.
	float avgDistance = 0.0f;
	int counter = 0;
	int incorrectCenterpoints = 0;
	for(int x = 0; x < extractedCenterlines->getWidth(); x++) {
	for(int y = 0; y < extractedCenterlines->getHeight(); y++) {
	for(int z = 0; z < extractedCenterlines->getDepth(); z++) {
		if(extractedCenterlines->get(x,y,z) == 1) {
			// Do BFS from this point on the centerlines volume to find closest real centerpoint
			int3 start(x,y,z);
			std::queue<int3> queue;
			unordered_set<int> visited;
			queue.push(start);
			bool firstFound = false;
			float bestDistance;
			int3 closest;

			while(!queue.empty()) {
				int3 current = queue.front();
				queue.pop();
				visited.insert(current.x+current.y*extractedCenterlines->getWidth()+current.z*extractedCenterlines->getWidth()*extractedCenterlines->getHeight());
				if(current.distance(start) > 5) // max distance for finding a closest centerpoint
					continue;

				if(realCenterlines->get(current) == 1) {
					float distance = current.distance(start);
					if(!firstFound) {
						firstFound = true;
						bestDistance = distance;
						closest = current;
					} else if(bestDistance > distance) {
						// found a better one
						bestDistance = distance;
						closest = current;
					} else if(distance > bestDistance+2) {
						// No more better can be found
						avgDistance += bestDistance;
						counter ++;
						detectedCenterlines->set(closest, 2); // mark centerpoint as detected
						break;
					}
				}

				// Add neighbors that are not visited
				for(int a = -1; a < 2; a++) {
				for(int b = -1; b < 2; b++) {
				for(int c = -1; c < 2; c++) {
					int3 n = current + int3(a,b,c);
					if(extractedCenterlines->inBounds(n) && visited.find(n.x+n.y*extractedCenterlines->getWidth()+z*extractedCenterlines->getWidth()*extractedCenterlines->getHeight()) == visited.end()) {
						queue.push(n);
					}

				}}}
			}

			if(!firstFound)
				incorrectCenterpoints++;
		}
	}}}
	avgDistance /= counter;
	result.averageDistanceFromCenterline = avgDistance;

	std::cout << "Centerline result" << std::endl << "--------------------" << std::endl;
	std::cout << "Average distance from real centerline was: " << avgDistance << " voxels" << std::endl;
	std::cout << "Number of incorrect centerpoints: " << incorrectCenterpoints << std::endl;

	Volume<float3> * visualization = new Volume<float3>(detectedCenterlines->getSize());
	for(int i = 0; i < visualization->getTotalSize(); i++) {
		if(detectedCenterlines->get(i) == 2) {
			float3 v;
			v.x = 1.0;
			v.y = 0.0;
			v.z = 0.0;
			visualization->set(i, v);
		}
	}

	// voxels with detectedCenterlines == 2 are the ones that are detected, == 1 are those that are not detected


	std::vector<int3> detectedPoints;

	for(int x = 0; x < extractedCenterlines->getWidth(); x++) {
	for(int y = 0; y < extractedCenterlines->getHeight(); y++) {
	for(int z = 0; z < extractedCenterlines->getDepth(); z++) {
		int3 current(x,y,z);

		if(detectedCenterlines->get(current) == 2)
			detectedPoints.push_back(current);
	}}}

	// Fill gaps
	const int maxDistance = 4;
	const int width = extractedCenterlines->getWidth();
	const int height = extractedCenterlines->getHeight();
	for(int i = 0; i < detectedPoints.size(); i++) {
	for(int j = 0; j < i; j++) {
		// For each pair of detected points see if there are any undetected points nearby
		// Do a flood fill from detected point i and see if j is found. If j is found add all undetected points nearby
		float distance = detectedPoints[i].distance(detectedPoints[j]);
		if(distance < maxDistance) {
			int3 direction = detectedPoints[j]-detectedPoints[i];

			std::queue<int3> queue;
			queue.push(detectedPoints[i]);
			std::vector<int3> undetected;
			unordered_set<int> visited;
			bool jFound = true;
			while(!queue.empty()) {
				int3 current = queue.front();
				queue.pop();

				if(current.distance(detectedPoints[i]) > maxDistance)
					continue;

				// If point is undetected, add to undetected
				if(detectedCenterlines->get(current) == 1 &&
					((current-detectedPoints[i]).normalize()).dot((detectedPoints[j]-current).normalize()) > -0.01) {
					undetected.push_back(current);
				}

				// If point is j, set jFound and break loop
				if(current == detectedPoints[j]) {
					jFound = true;
					break;
				}

				// Add neighbors if in right direction
				const int nSize = 1;
				for(int a = -nSize; a < nSize+1; a++) {
				for(int b = -nSize; b < nSize+1; b++) {
				for(int c = -nSize; c < nSize+1; c++) {
					int3 n = current + int3(a,b,c);
					//int3 c = n-detectedPoints[i];
					if(detectedCenterlines->inBounds(n) &&
							visited.find(n.x+n.y*width+n.z*width*height) == visited.end()) {
						queue.push(n);
						visited.insert(current.x+current.y*width+current.z*width*height);
					}
				}}}
			}

			if(jFound) {
				for(int k = 0; k < undetected.size(); k++) {
					detectedCenterlines->set(undetected[k], 2);
				}
			}
		}
	}}

	// Of the marked centerlines, fill small gaps, and measure percentage of extracted tree
	int undetected = 0;
	int detected = 0;
	for(int i = 0; i < detectedCenterlines->getTotalSize(); i++) {
		if(detectedCenterlines->get(i) == 1) {
			undetected++;
			float3 v = visualization->get(i);
			v.z = 1.0f;
			visualization->set(i, v);
		} else if(detectedCenterlines->get(i) == 2) {
			float3 v = visualization->get(i);
			// is it newly detected?
			if(v.x != 1.0) {
				v.y = 1.0;
			}
			//v.z = (float)noisy->get(i)/255.0f;
			visualization->set(i, v);
			detected++;
		}
	}

	//visualization->showMIP();

	result.percentageExtractedCenterlines = (float)detected*100.0f/(detected+undetected);
	std::cout << "Percentage of extracted centerline: " << result.percentageExtractedCenterlines << std::endl << std::endl;


	int extracted = 0;
	for(int i = 0; i < extractedCenterlines->getTotalSize(); i++) {
		if(extractedCenterlines->get(i) == 1)
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
	Volume<char> * segmentation = new Volume<char>(*(output->getSize()));
	segmentation->setData(output->getSegmentation());
	for(int i = 0; i < original->getTotalSize(); i++) {
		bool truth = original->get(i) > 0;
		bool test = segmentation->get(i) > 0;
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

	float recall = (float)truePositives / (truePositives+falseNegatives);
	float precision = (float)truePositives / (truePositives+falsePositives);
	std::cout << "Recall: " << recall << std::endl;
	std::cout << "Precision: " << precision << std::endl;

	result.recall = recall;
	result.precision = precision;

	return result;

}
