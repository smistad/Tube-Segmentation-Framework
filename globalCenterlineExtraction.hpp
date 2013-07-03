#ifndef GLOBAL_CENTERLINE_EXTRACTION_H
#define GLOBAL_CENTERLINE_EXTRACTION_H
#include "SIPL/Types.hpp"
#include <vector>
#include "tube-segmentation.hpp"
using namespace SIPL;

class CrossSection {
public:
	int3 pos;
	float TDF;
	std::vector<CrossSection *> neighbors;
	int label;
	int index;
	float3 direction;
};

class Connection;

class Segment {
public:
	std::vector<CrossSection *> sections;
	std::vector<Connection *> connections;
	float benefit;
	float cost;
	int index;
};

class Connection {
public:
	Segment * source;
	Segment * target;
	CrossSection * source_section;
	CrossSection * target_section;
	float cost;
};

std::vector<CrossSection *> createGraph(TubeSegmentation &T, SIPL::int3 size);

std::vector<Segment *> createSegments(OpenCL &ocl, TubeSegmentation &TS, std::vector<CrossSection *> &crossSections, SIPL::int3 size);
int selectRoot(std::vector<Segment *> segments);
int * createDepthFirstOrdering(std::vector<Segment *> segments, int root, int &Ns);

std::vector<Segment *> minimumSpanningTree(Segment * root, int3 size);
std::vector<Segment *> findOptimalSubtree(std::vector<Segment *> segments, int * depthFirstOrdering, int Ns);
void createConnections(TubeSegmentation &TS, std::vector<Segment *> segments, int3 size);
#endif
