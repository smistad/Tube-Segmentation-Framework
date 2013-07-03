#include "parallelCenterlineExtraction.hpp"
#include "tube-segmentation.hpp"
#include <vector>
#include <queue>
#include "inputOutput.hpp"
#include "histogram-pyramids.hpp"
#include "eigenanalysisOfHessian.hpp"
#ifdef CPP11
#include <unordered_set>
using std::unordered_set;
#else
#include <boost/unordered_set.hpp>
using boost::unordered_set;
#endif
using namespace cl;
using namespace SIPL;
#define MAX(a,b) a > b ? a : b

#define LPOS(a,b,c) (a)+(b)*(size.x)+(c)*(size.x*size.y)
#define POS(pos) pos.x+pos.y*size.x+pos.z*size.x*size.y
#define M(a,b,c) 1-sqrt(pow(T.Fx[a+b*size.x+c*size.x*size.y],2.0f) + pow(T.Fy[a+b*size.x+c*size.x*size.y],2.0f) + pow(T.Fz[a+b*size.x+c*size.x*size.y],2.0f))
#define SQR_MAG(pos) sqrt(pow(T.Fx[pos.x+pos.y*size.x+pos.z*size.x*size.y],2.0f) + pow(T.Fy[pos.x+pos.y*size.x+pos.z*size.x*size.y],2.0f) + pow(T.Fz[pos.x+pos.y*size.x+pos.z*size.x*size.y],2.0f))
#define SQR_MAG_SMALL(pos) sqrt(pow(T.FxSmall[pos.x+pos.y*size.x+pos.z*size.x*size.y],2.0f) + pow(T.FySmall[pos.x+pos.y*size.x+pos.z*size.x*size.y],2.0f) + pow(T.FzSmall[pos.x+pos.y*size.x+pos.z*size.x*size.y],2.0f))


class Edge;
class Node {
public:
	int3 pos;
	std::vector<Edge *> edges;

};

class Edge {
public:
	Node * source;
	Node * target;
	std::vector<Node *> removedNodes;
	float distance;
};

class EdgeComparator {
public:
	bool operator()(Edge * a, Edge *b) const {
		return a->distance > b->distance;
	}
};


void removeEdge(Node * n, Node * remove) {
	std::vector<Edge *> edges;
	for(int i = 0; i < n->edges.size(); i++) {
		Edge * e = n->edges[i];
		if(e->target != remove) {
			edges.push_back(e);
		}
	}
	n->edges = edges;
}

void restoreNodes(Edge * e, std::vector<Node *> &finalGraph) {
	if(e->removedNodes.size() == 0)
		return;

	// Remove e from e->source
	removeEdge(e->source, e->target);
	Node * previous = e->source;
	for(int k = 0; k < e->removedNodes.size(); k++) {
		Edge * newEdge = new Edge;
		// Create edge from source to k or k-1 to k
		newEdge->source = previous;
		newEdge->target = e->removedNodes[k];
		previous->edges.push_back(newEdge);
		previous = e->removedNodes[k];
		finalGraph.push_back(e->removedNodes[k]);

	}
	// Create edge from last k to target
	Edge * newEdge = new Edge;
	newEdge->source = previous;
	newEdge->target = e->target;
	previous->edges.push_back(newEdge);
}

std::vector<Node *> minimumSpanningTreePCE(
		int root,
		std::vector<Node *> &graph,
		int3 &size,
		unordered_set<int> &visited
		) {
	std::vector<Node *> result;
	std::priority_queue<Edge *, std::vector<Edge *>, EdgeComparator> queue;

	// Start with graph[0]
	result.push_back(graph[root]);
	visited.insert(POS(graph[root]->pos));

	// Add edges to priority queue
	for(int i = 0; i < graph[root]->edges.size(); i++) {
		Edge * en = graph[root]->edges[i];
		queue.push(en);
	}
	graph[root]->edges.clear();

	while(!queue.empty()) {
		Edge * e = queue.top();
		queue.pop();

		if(visited.find(POS(e->target->pos)) != visited.end())
			continue; // already visited

		// Add all edges of e->target to queue if targets have not been added
		for(int i = 0; i < e->target->edges.size(); i++) {
			Edge * en = e->target->edges[i];
			if(visited.find(POS(en->target->pos)) == visited.end()) {
				queue.push(en);
			}
		}

		e->target->edges.clear(); // Remove all edges first
		e->source->edges.push_back(e); // Add edge to source
		result.push_back(e->target); // Add target to result
		visited.insert(POS(e->target->pos));
	}

	return result;
}

void removeLoops(
		std::vector<int3> &vertices,
		std::vector<SIPL::int2> &edges,
		int3 &size
	) {

	std::vector<Node *> nodes;
	for(int i = 0; i < vertices.size(); i++) {
		Node * n = new Node;
		n->pos = vertices[i];
		nodes.push_back(n);
	}
	for(int i = 0; i < edges.size(); i++) {
		Node * a = nodes[edges[i].x];
		Node * b = nodes[edges[i].y];
		Edge * e = new Edge;
		e->distance = a->pos.distance(b->pos);
		e->source = a;
		e->target = b;
		a->edges.push_back(e);

		Edge * e2 = new Edge;
		e2->distance = a->pos.distance(b->pos);
		e2->source = b;
		e2->target = a;
		b->edges.push_back(e2);
	}

	// Create graph
	std::vector<Node *> graph;

	// Remove all nodes with degree 2 and add them to edge
	for(int i = 0; i < nodes.size(); i++) {
		Node * n = nodes[i];
		if(n->edges.size() == 2) {
			// calculate distance
			float distance = n->edges[0]->distance+n->edges[1]->distance;
			Node * a = n->edges[0]->target;
			Node * b = n->edges[1]->target;
			// Fuse the two nodes together
			Edge * e = new Edge;
			e->source = a;
			e->target = b;
			e->distance = distance;
			// Add removed nodes from previous edges
			for(int j = n->edges[0]->removedNodes.size()-1; j >= 0; j--) {
				e->removedNodes.push_back(n->edges[0]->removedNodes[j]);
			}
			e->removedNodes.push_back(n);
			for(int j = 0; j < n->edges[1]->removedNodes.size(); j++) {
				e->removedNodes.push_back(n->edges[1]->removedNodes[j]);
			}

			Edge * e2 = new Edge;
			e2->source = b;
			e2->target = a;
			e2->distance = distance;
			// Add removed nodes from previous edges
			for(int j = n->edges[1]->removedNodes.size()-1; j >= 0; j--) {
				e2->removedNodes.push_back(n->edges[1]->removedNodes[j]);
			}
			e2->removedNodes.push_back(n);
			for(int j = 0; j < n->edges[0]->removedNodes.size(); j++) {
				e2->removedNodes.push_back(n->edges[0]->removedNodes[j]);
			}

			removeEdge(a,n);
			removeEdge(b,n);
			n->edges.clear();
			a->edges.push_back(e);
			b->edges.push_back(e2);
		} else {
			// add node to graph
			graph.push_back(n);
		}
	}

	// Do MST with edge distance as cost
	int sizeBeforeMST = graph.size();
	if(sizeBeforeMST == 0) {
	    throw SIPL::SIPLException("Centerline graph size is 0! Can't continue. Maybe lower min-tree-length?", __LINE__,__FILE__);
	}
	unordered_set<int> visited;
	std::vector<Node *> newGraph = minimumSpanningTreePCE(0, graph, size, visited);
	int sizeAfterMST = newGraph.size();

	while(newGraph.size() < graph.size()) {
		// Some nodes were not visited: find new root
		int root;
		for(int i = 0; i < graph.size(); i++) {
			if(visited.find(POS(graph[i]->pos)) == visited.end()) {
				// i has not been used
				root = i;
			}
		}

		std::vector<Node *> newGraph2 = minimumSpanningTreePCE(root, graph, size, visited);
		for(int i = 0; i < newGraph2.size(); i++) {
			newGraph.push_back(newGraph2[i]);
		}
		int sizeAfterMST = newGraph.size();
	}

	// Restore graph
	// For all edges that are in the MST graph: get nodes that was on these edges
	std::vector<Node *> finalGraph;
	for(int i = 0; i < newGraph.size(); i++) {
		Node * n = newGraph[i];
		finalGraph.push_back(n);
		std::vector<Edge *> nEdges = n->edges;
		for(int j = 0; j < nEdges.size(); j++) {
			Edge * e = nEdges[j];
			restoreNodes(e, finalGraph);
		}
	}


	std::vector<int3> newVertices;
	std::vector<SIPL::int2> newEdges;
	unordered_map<int, int> added;
	int counter = 0;
	for(int i = 0; i < finalGraph.size(); i++) {
		Node * n = finalGraph[i];
		int nIndex;
		if(added.find(POS(n->pos)) != added.end()) {
			// already has been added
			// fetch index
			nIndex = added[POS(n->pos)];
		} else {
			newVertices.push_back(n->pos);
			added[POS(n->pos)] = counter;
			nIndex = counter;
			counter++;
		}
		for(int j = 0; j < n->edges.size(); j++) {
			Edge * e = n->edges[j];
			// add edge from n to n_j

			int tIndex;
			if(added.find(POS(e->target->pos)) != added.end()) {
				// already has been added
				// fetch index
				tIndex = added[POS(e->target->pos)];
			} else {
				newVertices.push_back(e->target->pos);
				added[POS(e->target->pos)] = counter;
				tIndex = counter;
				counter++;
			}
			newEdges.push_back(SIPL::int2(nIndex, tIndex));
		}
	}

	// Cleanup
	vertices = newVertices;
	edges = newEdges;
}

char * createCenterlineVoxels(
		std::vector<int3> &vertices,
		std::vector<SIPL::int2> &edges,
		float * radius,
		int3 &size
		) {
	const int totalSize = size.x*size.y*size.z;
	char * centerlines = new char[totalSize]();

	for(int i = 0; i < edges.size(); i++) {
		int3 a = vertices[edges[i].x];
		int3 b = vertices[edges[i].y];
		float distance = a.distance(b);
		float3 direction(b.x-a.x,b.y-a.y,b.z-a.z);
		int n = ceil(distance);
		float avgRadius = 0.0f;
		for(int j = 0; j < n; j++) {
			float ratio = (float)j/n;
			float3 pos = a+direction*ratio;
			int3 intPos(round(pos.x), round(pos.y), round(pos.z));
			centerlines[POS(intPos)] = 1;
			avgRadius += radius[POS(intPos)];
		}
		avgRadius /= n;

		for(int j = 0; j < n; j++) {
			float ratio = (float)j/n;
			float3 pos = a+direction*ratio;
			int3 intPos(round(pos.x), round(pos.y), round(pos.z));
			radius[POS(intPos)] = avgRadius;
		}

	}

	return centerlines;
}
Image3D runNewCenterlineAlgWithoutOpenCL(OpenCL &ocl, SIPL::int3 size, paramList &parameters, Image3D &vectorField, Image3D &TDF, Image3D &radius) {
    const int totalSize = size.x*size.y*size.z;
	const bool no3Dwrite = !getParamBool(parameters, "3d_write");
    const int cubeSize = getParam(parameters, "cube-size");
    const int minTreeLength = getParam(parameters, "min-tree-length");
    const float Thigh = getParam(parameters, "tdf-high");
    const float minAvgTDF = getParam(parameters, "min-mean-tdf");
    const float maxDistance = getParam(parameters, "max-distance");

    cl::size_t<3> offset;
    offset[0] = 0;
    offset[1] = 0;
    offset[2] = 0;
    cl::size_t<3> region;
    region[0] = size.x;
    region[1] = size.y;
    region[2] = size.z;

    // Transfer TDF, vectorField and radius to host
    TubeSegmentation T;
    T.Fx = new float[totalSize];
    T.Fy = new float[totalSize];
    T.Fz = new float[totalSize];
    T.TDF = new float[totalSize];

    if(!getParamBool(parameters, "16bit-vectors")) {
    	// 32 bit vector fields
        float * Fs = new float[totalSize*4];
        ocl.queue.enqueueReadImage(vectorField, CL_TRUE, offset, region, 0, 0, Fs);
#pragma omp parallel for
        for(int i = 0; i < totalSize; i++) {
            T.Fx[i] = Fs[i*4];
            T.Fy[i] = Fs[i*4+1];
            T.Fz[i] = Fs[i*4+2];
        }
        delete[] Fs;
        ocl.queue.enqueueReadImage(TDF, CL_TRUE, offset, region, 0, 0, T.TDF);
    } else {
    	// 16 bit vector fields
        short * Fs = new short[totalSize*4];
        ocl.queue.enqueueReadImage(vectorField, CL_TRUE, offset, region, 0, 0, Fs);
#pragma omp parallel for
        for(int i = 0; i < totalSize; i++) {
            T.Fx[i] = MAX(-1.0f, Fs[i*4] / 32767.0f);
            T.Fy[i] = MAX(-1.0f, Fs[i*4+1] / 32767.0f);;
            T.Fz[i] = MAX(-1.0f, Fs[i*4+2] / 32767.0f);
        }
        delete[] Fs;

        // Convert 16 bit TDF to 32 bit
        unsigned short * tempTDF = new unsigned short[totalSize];
        ocl.queue.enqueueReadImage(TDF, CL_TRUE, offset, region, 0, 0, tempTDF);
#pragma omp parallel for
        for(int i = 0; i < totalSize; i++) {
            T.TDF[i] = (float)tempTDF[i] / 65535.0f;
        }
        delete[] tempTDF;
    }
    T.radius = new float[totalSize];
    ocl.queue.enqueueReadImage(radius, CL_TRUE, offset, region, 0, 0, T.radius);

    // Get candidate points
    std::vector<int3> candidatePoints;
#pragma omp parallel for
    for(int z = 1; z < size.z-1; z++) {
    for(int y = 1; y < size.y-1; y++) {
    for(int x = 1; x < size.x-1; x++) {
       int3 pos(x,y,z);
       if(T.TDF[POS(pos)] >= Thigh) {
#pragma omp critical
           candidatePoints.push_back(pos);
       }
    }}}
    std::cout << "candidate points: " << candidatePoints.size() << std::endl;

    unordered_set<int> filteredPoints;
#pragma omp parallel for
    for(int i = 0; i < candidatePoints.size(); i++) {
        int3 pos = candidatePoints[i];
        // Filter candidate points
        const float thetaLimit = 0.5f;
        const float radii = T.radius[POS(pos)];
        const int maxD = std::max(std::min((float)round(radii), 5.0f), 1.0f);
        bool invalid = false;

        float3 e1 = getTubeDirection(T, pos, size);

        for(int a = -maxD; a <= maxD; a++) {
        for(int b = -maxD; b <= maxD; b++) {
        for(int c = -maxD; c <= maxD; c++) {
            if(a == 0 && b == 0 && c == 0)
                continue;
            float3 r(a,b,c);
            float length = r.length();
            int3 n(pos.x + r.x,pos.y+r.y,pos.z+r.z);
            if(!inBounds(n,size))
                continue;
            float dp = e1.dot(r);
            float3 r_projected(r.x-e1.x*dp,r.y-e1.y*dp,r.z-e1.z*dp);
            float3 rn = r.normalize();
            float3 r_projected_n = r_projected.normalize();
            float theta = acos(rn.dot(r_projected_n));
            if((theta < thetaLimit && length < maxD)) {
                if(SQR_MAG(n) < SQR_MAG(pos)) {
                    invalid = true;
                    break;
                }

            }
        }}}
        if(!invalid) {
#pragma omp critical
            filteredPoints.insert(POS(pos));
        }
    }
    candidatePoints.clear();
    std::cout << "filtered points: " << filteredPoints.size() << std::endl;

    std::vector<int3> centerpoints;
    for(int z = 0; z < size.z/cubeSize; z++) {
    for(int y = 0; y < size.y/cubeSize; y++) {
    for(int x = 0; x < size.x/cubeSize; x++) {
        int3 bestPos;
        float bestTDF = 0.0f;
        int3 readPos(
            x*cubeSize,
            y*cubeSize,
            z*cubeSize
        );
        bool found = false;
        for(int a = 0; a < cubeSize; a++) {
        for(int b = 0; b < cubeSize; b++) {
        for(int c = 0; c < cubeSize; c++) {
            int3 pos = readPos + int3(a,b,c);
            if(filteredPoints.find(POS(pos)) != filteredPoints.end()) {
                float tdf = T.TDF[POS(pos)];
                if(tdf > bestTDF) {
                    found = true;
                    bestTDF = tdf;
                    bestPos = pos;
                }
            }
        }}}
        if(found) {
#pragma omp critical
            centerpoints.push_back(bestPos);
        }
    }}}

    int nofPoints = centerpoints.size();
    std::cout << "filtered points: " <<nofPoints<< std::endl;
    std::vector<SIPL::int2> edges;

    // Do linking
    for(int i = 0; i < nofPoints;i++) {
        int3 xa = centerpoints[i];
        SIPL::int2 bestPair;
        float shortestDistance = maxDistance*2;
        bool validPairFound = false;

        for(int j = 0; j < nofPoints;j++) {
            if(i == j)
                continue;
            int3 xb = centerpoints[j];

            int db = round(xa.distance(xb));
            if(db >= shortestDistance)
                continue;
            for(int k = 0; k < j;k++) {
                if(k == i)
                    continue;
                int3 xc = centerpoints[k];

                int dc = round(xa.distance(xc));

                if(db+dc < shortestDistance) {
                    // Check angle
                    int3 ab = (xb-xa);
                    int3 ac = (xc-xa);
                    float angle = acos(ab.normalize().dot(ac.normalize()));
                    //printf("angle: %f\n", angle);
                    if(angle < 2.0f) // 120 degrees
                    //if(angle < 1.57f) // 90 degrees
                        continue;

                    // Check avg TDF for a-b
                    float avgTDF = 0.0f;
                    for(int l = 0; l <= db; l++) {
                        float alpha = (float)l/db;
                        int3 p((int)round(xa.x+ab.x*alpha),(int)round(xa.y+ab.y*alpha),(int)round(xa.z+ab.z*alpha));
                        float t = T.TDF[POS(p)];
                        avgTDF += t;
                    }
                    avgTDF /= db+1;
                    if(avgTDF < minAvgTDF)
                        continue;

                    avgTDF = 0.0f;

                    // Check avg TDF for a-c
                    for(int l = 0; l <= dc; l++) {
                        float alpha = (float)l/dc;
                        int3 p((int)round(xa.x+ac.x*alpha),(int)round(xa.y+ac.y*alpha),(int)round(xa.z+ac.z*alpha));
                        float t = T.TDF[POS(p)];
                        avgTDF += t;
                    }
                    avgTDF /= dc+1;

                    if(avgTDF < minAvgTDF)
                        continue;

                    validPairFound = true;
                    bestPair.x = j;
                    bestPair.y = k;
                    shortestDistance = db+dc;
                }
            } // k
        }// j

        if(validPairFound) {
            // Store edges
            SIPL::int2 edge(i, bestPair.x);
            SIPL::int2 edge2(i, bestPair.y);
            edges.push_back(edge);
            edges.push_back(edge2);
        }
    } // i
    std::cout << "nr of edges: " << edges.size() << std::endl;

    // Do graph component labeling
    // Create initial labels
    int * labels = new int[nofPoints];
    for(int i = 0; i < nofPoints; i++) {
        labels[i] = i;
    }

    // Do iteratively using edges until no more changes
    bool changeDetected = true;
    while(changeDetected) {
        changeDetected = false;
        for(int i = 0; i < edges.size(); i++) {
            SIPL::int2 edge = edges[i];
            if(labels[edge.x] != labels[edge.y]) {
                changeDetected = true;
                if(labels[edge.x] < labels[edge.y]) {
                    labels[edge.x] = labels[edge.y];
                } else {
                    labels[edge.y] = labels[edge.x];
                }
            }
        }
    }


    // Calculate length of each label
    int * lengths = new int[nofPoints]();
    for(int i = 0; i < nofPoints; i++) {
        lengths[labels[i]]++;
    }
    std::vector<int3> vertices = centerpoints;

    // Select wanted parts of centerline

    std::vector<SIPL::int2> edges2;
    int counter = nofPoints;
    int maxEdgeDistance = getParam(parameters, "max-edge-distance");
    for(int i = 0; i < edges.size(); i++) {
        if(lengths[labels[edges[i].x]] >= minTreeLength && lengths[labels[edges[i].y]] >= minTreeLength ) {
            // Check length of edge
            int3 A = vertices[edges[i].x];
            int3 B = vertices[edges[i].y];
            float distance = A.distance(B);
            if(getParamStr(parameters, "centerline-vtk-file") != "off" &&
                    distance > maxEdgeDistance) {
                float3 direction(B.x-A.x,B.y-A.y,B.z-A.z);
                float3 Af(A.x,A.y,A.z);
                int previous = edges[i].x;
                for(int j = maxEdgeDistance; j < distance; j += maxEdgeDistance) {
                    float3 newPos = Af + ((float)j/distance)*direction;
                    int3 newVertex(round(newPos.x), round(newPos.y), round(newPos.z));
                    // Create new vertex
                    vertices.push_back(newVertex);
                    // Add new edge
                    SIPL::int2 edge(previous, counter);
                    edges2.push_back(edge);
                    previous = counter;
                    counter++;
                }
                // Connect previous vertex to B
                SIPL::int2 edge(previous, edges[i].y);
                edges2.push_back(edge);
            } else {
                edges2.push_back(edges[i]);
            }
        }
    }
    edges = edges2;

    // Remove loops from graph
    if(getParamBool(parameters, "loop-removal"))
        removeLoops(vertices, edges, size);

    ocl.queue.finish();
    char * centerlinesData = createCenterlineVoxels(vertices, edges, T.radius, size);
    Image3D centerlines= Image3D(
        ocl.context,
        CL_MEM_READ_WRITE,
        ImageFormat(CL_R, CL_SIGNED_INT8),
        size.x, size.y, size.z
    );

    ocl.queue.enqueueWriteImage(
            centerlines,
            CL_FALSE,
            offset,
            region,
            0, 0,
            centerlinesData
    );
    ocl.queue.enqueueWriteImage(
            radius,
            CL_FALSE,
            offset,
            region,
            0, 0,
            T.radius
    );

    if(getParamStr(parameters, "centerline-vtk-file") != "off") {
        writeToVtkFile(parameters, vertices, edges);
    }

    ocl.queue.finish();

    delete[] T.TDF;
    delete[] T.Fx;
    delete[] T.Fy;
    delete[] T.Fz;
    delete[] T.radius;
    delete[] centerlinesData;

    return centerlines;
}

Image3D runNewCenterlineAlg(OpenCL &ocl, SIPL::int3 size, paramList &parameters, Image3D &vectorField, Image3D &TDF, Image3D &radius) {
    if(ocl.platform.getInfo<CL_PLATFORM_VENDOR>().substr(0,5) == "Apple") {
        std::cout << "Apple platform detected. Running centerline extraction without OpenCL." << std::endl;
        return runNewCenterlineAlgWithoutOpenCL(ocl,size,parameters,vectorField,TDF,radius);
    }
    const int totalSize = size.x*size.y*size.z;
	const bool no3Dwrite = !getParamBool(parameters, "3d_write");
    const int cubeSize = getParam(parameters, "cube-size");
    const int minTreeLength = getParam(parameters, "min-tree-length");
    const float Thigh = getParam(parameters, "tdf-high");
    const float Tmean = getParam(parameters, "min-mean-tdf");
    const float maxDistance = getParam(parameters, "max-distance");

    cl::size_t<3> offset;
    offset[0] = 0;
    offset[1] = 0;
    offset[2] = 0;
    cl::size_t<3> region;
    region[0] = size.x;
    region[1] = size.y;
    region[2] = size.z;

    Kernel candidatesKernel(ocl.program, "findCandidateCenterpoints");
    Kernel candidates2Kernel(ocl.program, "findCandidateCenterpoints2");
    Kernel ddKernel(ocl.program, "dd");
    Kernel initCharBuffer(ocl.program, "initCharBuffer");

    cl::Event startEvent, endEvent;
    cl_ulong start, end;
    if(getParamBool(parameters, "timing")) {
        ocl.queue.enqueueMarker(&startEvent);
    }
    Image3D * centerpointsImage2 = new Image3D(
            ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_R, CL_SIGNED_INT8),
            size.x, size.y, size.z
    );
    ocl.GC.addMemObject(centerpointsImage2);
    Buffer vertices;
    int sum = 0;

    if(no3Dwrite) {
        Buffer * centerpoints = new Buffer(
                ocl.context,
                CL_MEM_READ_WRITE,
                sizeof(char)*totalSize
        );
        ocl.GC.addMemObject(centerpoints);

        candidatesKernel.setArg(0, TDF);
        candidatesKernel.setArg(1, *centerpoints);
        candidatesKernel.setArg(2, Thigh);
        ocl.queue.enqueueNDRangeKernel(
                candidatesKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NullRange
        );

        HistogramPyramid3DBuffer hp3(ocl);
        hp3.create(*centerpoints, size.x, size.y, size.z);

        candidates2Kernel.setArg(0, TDF);
        candidates2Kernel.setArg(1, radius);
        candidates2Kernel.setArg(2, vectorField);
        Buffer * centerpoints2 = new Buffer(
                ocl.context,
                CL_MEM_READ_WRITE,
                sizeof(char)*totalSize
        );
        ocl.GC.addMemObject(centerpoints2);
        initCharBuffer.setArg(0, *centerpoints2);
        ocl.queue.enqueueNDRangeKernel(
                initCharBuffer,
                NullRange,
                NDRange(totalSize),
                NullRange
        );

        candidates2Kernel.setArg(3, *centerpoints2);
        std::cout << "candidates: " << hp3.getSum() << std::endl;
        if(hp3.getSum() <= 0 || hp3.getSum() > 0.5*totalSize) {
        	throw SIPL::SIPLException("The number of candidate voxels is too low or too high. Something went wrong... Wrong parameters? Out of memory?", __LINE__, __FILE__);
        }
        hp3.traverse(candidates2Kernel, 4);
        ocl.queue.finish();
        hp3.deleteHPlevels();
        ocl.GC.deleteMemObject(centerpoints);
        ocl.queue.enqueueCopyBufferToImage(
            *centerpoints2,
            *centerpointsImage2,
            0,
            offset,
            region
        );
        ocl.queue.finish();
        ocl.GC.deleteMemObject(centerpoints2);

		if(getParamBool(parameters, "centerpoints-only")) {
			return *centerpointsImage2;
		}
        ddKernel.setArg(0, TDF);
        ddKernel.setArg(1, *centerpointsImage2);
        ddKernel.setArg(3, cubeSize);
        Buffer * centerpoints3 = new Buffer(
                ocl.context,
                CL_MEM_READ_WRITE,
                sizeof(char)*totalSize
        );
        ocl.GC.addMemObject(centerpoints3);
        initCharBuffer.setArg(0, *centerpoints3);
        ocl.queue.enqueueNDRangeKernel(
                initCharBuffer,
                NullRange,
                NDRange(totalSize),
                NullRange
        );
        ddKernel.setArg(2, *centerpoints3);
        ocl.queue.enqueueNDRangeKernel(
                ddKernel,
                NullRange,
                NDRange(ceil((float)size.x/cubeSize),ceil((float)size.y/cubeSize),ceil((float)size.z/cubeSize)),
                NullRange
        );
        ocl.queue.finish();
        ocl.GC.deleteMemObject(centerpointsImage2);

        // Construct HP of centerpointsImage
        HistogramPyramid3DBuffer hp(ocl);
        hp.create(*centerpoints3, size.x, size.y, size.z);
        sum = hp.getSum();
        std::cout << "number of vertices detected " << sum << std::endl;

        // Run createPositions kernel
        vertices = hp.createPositionBuffer();
        ocl.queue.finish();
        hp.deleteHPlevels();
        ocl.GC.deleteMemObject(centerpoints3);
    } else {
        Kernel init3DImage(ocl.program, "init3DImage");
        init3DImage.setArg(0, *centerpointsImage2);
        ocl.queue.enqueueNDRangeKernel(
            init3DImage,
            NullRange,
            NDRange(size.x,size.y,size.z),
            NullRange
        );

        Image3D * centerpointsImage = new Image3D(
                ocl.context,
                CL_MEM_READ_WRITE,
                ImageFormat(CL_R, CL_SIGNED_INT8),
                size.x, size.y, size.z
        );
        ocl.GC.addMemObject(centerpointsImage);

        candidatesKernel.setArg(0, TDF);
        candidatesKernel.setArg(1, *centerpointsImage);
        candidatesKernel.setArg(2, Thigh);
        ocl.queue.enqueueNDRangeKernel(
                candidatesKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NDRange(4,4,4)
        );


        candidates2Kernel.setArg(0, TDF);
        candidates2Kernel.setArg(1, radius);
        candidates2Kernel.setArg(2, vectorField);

        HistogramPyramid3D hp3(ocl);
        hp3.create(*centerpointsImage, size.x, size.y, size.z);
        std::cout << "candidates: " << hp3.getSum() << std::endl;
		if(hp3.getSum() <= 0 || hp3.getSum() > 0.5*totalSize) {
        	throw SIPL::SIPLException("The number of candidate voxels is too or too high. Something went wrong... Wrong parameters? Out of memory?", __LINE__, __FILE__);
        }

        candidates2Kernel.setArg(3, *centerpointsImage2);
        hp3.traverse(candidates2Kernel, 4);
        ocl.queue.finish();
        hp3.deleteHPlevels();
        ocl.GC.deleteMemObject(centerpointsImage);

        Image3D * centerpointsImage3 = new Image3D(
                ocl.context,
                CL_MEM_READ_WRITE,
                ImageFormat(CL_R, CL_SIGNED_INT8),
                size.x, size.y, size.z
        );
        ocl.GC.addMemObject(centerpointsImage3);
        init3DImage.setArg(0, *centerpointsImage3);
        ocl.queue.enqueueNDRangeKernel(
            init3DImage,
            NullRange,
            NDRange(size.x,size.y,size.z),
            NullRange
        );

		if(getParamBool(parameters, "centerpoints-only")) {
			return *centerpointsImage2;
		}
        ddKernel.setArg(0, TDF);
        ddKernel.setArg(1, *centerpointsImage2);
        ddKernel.setArg(3, cubeSize);
        ddKernel.setArg(2, *centerpointsImage3);
        ocl.queue.enqueueNDRangeKernel(
                ddKernel,
                NullRange,
                NDRange(ceil((float)size.x/cubeSize),ceil((float)size.y/cubeSize),ceil((float)size.z/cubeSize)),
                NullRange
        );
        ocl.queue.finish();
        ocl.GC.deleteMemObject(centerpointsImage2);

        // Construct HP of centerpointsImage
        HistogramPyramid3D hp(ocl);
        hp.create(*centerpointsImage3, size.x, size.y, size.z);
        sum = hp.getSum();
        std::cout << "number of vertices detected " << sum << std::endl;

        // Run createPositions kernel
        vertices = hp.createPositionBuffer();
        ocl.queue.finish();
        hp.deleteHPlevels();
        ocl.GC.deleteMemObject(centerpointsImage3);
    }
    if(sum < 8) {
    	throw SIPL::SIPLException("Too few centerpoints detected. Revise parameters.", __LINE__, __FILE__);
    } else if(sum >= 16384) {
    	throw SIPL::SIPLException("Too many centerpoints detected. More cropping of dataset is probably needed.", __LINE__, __FILE__);
    }

if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&endEvent);
    ocl.queue.finish();
    startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    endEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &end);
    std::cout << "RUNTIME centerpoint extraction: " << (end-start)*1.0e-6 << " ms" << std::endl;
}

if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&startEvent);
}
    // Run linking kernel
    Image2D edgeTuples = Image2D(
            ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_R, CL_UNSIGNED_INT8),
            sum, sum
    );
    Kernel init2DImage(ocl.program, "init2DImage");
    init2DImage.setArg(0, edgeTuples);
    ocl.queue.enqueueNDRangeKernel(
        init2DImage,
        NullRange,
        NDRange(sum, sum),
        NullRange
    );
    int globalSize = sum;
    while(globalSize % 64 != 0) globalSize++;

    // Create lengths image
    Image2D * lengths = new Image2D(
            ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_R, CL_FLOAT),
            sum, sum
    );
    ocl.GC.addMemObject(lengths);

    // Run linkLengths kernel
    Kernel linkLengths(ocl.program, "linkLengths");
    linkLengths.setArg(0, vertices);
    linkLengths.setArg(1, *lengths);
    ocl.queue.enqueueNDRangeKernel(
            linkLengths,
            NullRange,
            NDRange(sum, sum),
            NullRange
    );

    // Create and init compacted_lengths image
    float * cl = new float[sum*sum*2]();
    Image2D * compacted_lengths = new Image2D(
            ocl.context,
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            ImageFormat(CL_RG, CL_FLOAT),
            sum, sum,
            0,
            cl
    );
    ocl.GC.addMemObject(compacted_lengths);
    delete[] cl;

    // Create and initialize incs buffer
    Buffer incs = Buffer(
            ocl.context,
            CL_MEM_READ_WRITE,
            sizeof(int)*sum
    );

    Kernel initIncsBuffer(ocl.program, "initIntBuffer");
    initIncsBuffer.setArg(0, incs);
    ocl.queue.enqueueNDRangeKernel(
        initIncsBuffer,
        NullRange,
        NDRange(sum),
        NullRange
    );

    // Run compact kernel
    Kernel compactLengths(ocl.program, "compact");
    compactLengths.setArg(0, *lengths);
    compactLengths.setArg(1, incs);
    compactLengths.setArg(2, *compacted_lengths);
    compactLengths.setArg(3, maxDistance);
    ocl.queue.enqueueNDRangeKernel(
            compactLengths,
            NullRange,
            NDRange(sum, sum),
            NullRange
    );
    ocl.queue.finish();
    ocl.GC.deleteMemObject(lengths);

    Kernel linkingKernel(ocl.program, "linkCenterpoints");
    linkingKernel.setArg(0, TDF);
    linkingKernel.setArg(1, vertices);
    linkingKernel.setArg(2, edgeTuples);
    linkingKernel.setArg(3, *compacted_lengths);
    linkingKernel.setArg(4, sum);
    linkingKernel.setArg(5, Tmean);
    linkingKernel.setArg(6, maxDistance);
    ocl.queue.enqueueNDRangeKernel(
            linkingKernel,
            NullRange,
            NDRange(globalSize),
            NDRange(64)
    );
    ocl.queue.finish();
    ocl.GC.deleteMemObject(compacted_lengths);
if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&endEvent);
    ocl.queue.finish();
    startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    endEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &end);
    std::cout << "RUNTIME linking: " << (end-start)*1.0e-6 << " ms" << std::endl;
}

if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&startEvent);
}

	// Remove duplicate edges
	Image2D edgeTuples2 = Image2D(
			ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_R, CL_UNSIGNED_INT8),
            sum, sum
    );
	Kernel removeDuplicatesKernel(ocl.program, "removeDuplicateEdges");
	removeDuplicatesKernel.setArg(0, edgeTuples);
	removeDuplicatesKernel.setArg(1, edgeTuples2);
	ocl.queue.enqueueNDRangeKernel(
			removeDuplicatesKernel,
			NullRange,
			NDRange(sum,sum),
			NullRange
	);
	edgeTuples = edgeTuples2;

    // Run HP on edgeTuples
    HistogramPyramid2D hp2(ocl);
    hp2.create(edgeTuples, sum, sum);

	std::cout << "number of edges detected " << hp2.getSum() << std::endl;
    if(hp2.getSum() == 0) {
        throw SIPL::SIPLException("No edges were found", __LINE__, __FILE__);
    } else if(hp2.getSum() > 10000000) {
    	throw SIPL::SIPLException("More than 10 million edges found. Must be wrong!", __LINE__, __FILE__);
    } else if(hp2.getSum() < 0){
    	throw SIPL::SIPLException("A negative number of edges was found!", __LINE__, __FILE__);
    }

    // Run create positions kernel on edges
    Buffer edges = hp2.createPositionBuffer();
    ocl.queue.finish();
    hp2.deleteHPlevels();
if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&endEvent);
    ocl.queue.finish();
    startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    endEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &end);
    std::cout << "RUNTIME HP creation and traversal: " << (end-start)*1.0e-6 << " ms" << std::endl;
}

if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&startEvent);
}

    // Do graph component labeling
    Buffer C = Buffer(
            ocl.context,
            CL_MEM_READ_WRITE,
            sizeof(int)*sum
    );
    Kernel initCBuffer(ocl.program, "initIntBufferID");
    initCBuffer.setArg(0, C);
    initCBuffer.setArg(1, sum);
    ocl.queue.enqueueNDRangeKernel(
        initCBuffer,
        NullRange,
        NDRange(globalSize),
        NDRange(64)
    );


    Buffer m = Buffer(
            ocl.context,
            CL_MEM_WRITE_ONLY,
            sizeof(int)
    );

    Kernel labelingKernel(ocl.program, "graphComponentLabeling");
    labelingKernel.setArg(0, edges);
    labelingKernel.setArg(1, C);
    labelingKernel.setArg(2, m);
    int M;
    int sum2 = hp2.getSum();
    labelingKernel.setArg(3, sum2);
    globalSize = sum2;
    while(globalSize % 64 != 0) globalSize++;
    int i = 0;
    int minIterations = 100;
    do {
        // write 0 to m
        if(i > minIterations) {
            M = 0;
            ocl.queue.enqueueWriteBuffer(m, CL_FALSE, 0, sizeof(int), &M);
        } else {
            M = 1;
        }

        ocl.queue.enqueueNDRangeKernel(
                labelingKernel,
                NullRange,
                NDRange(globalSize),
                NDRange(64)
        );

        // read m from device
        if(i > minIterations)
            ocl.queue.enqueueReadBuffer(m, CL_TRUE, 0, sizeof(int), &M);
        ++i;
    } while(M == 1);
    std::cout << "did graph component labeling in " << i << " iterations " << std::endl;
if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&endEvent);
    ocl.queue.finish();
    startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    endEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &end);
    std::cout << "RUNTIME graph component labeling: " << (end-start)*1.0e-6 << " ms" << std::endl;
}


if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&startEvent);
}
    // Remove small trees
    Buffer S = Buffer(
            ocl.context,
            CL_MEM_READ_WRITE,
            sizeof(int)*sum
    );
    Kernel initIntBuffer(ocl.program, "initIntBuffer");
    initIntBuffer.setArg(0, S);
    ocl.queue.enqueueNDRangeKernel(
        initIntBuffer,
        NullRange,
        NDRange(sum),
        NullRange
    );
    Kernel calculateTreeLengthKernel(ocl.program, "calculateTreeLength");
    calculateTreeLengthKernel.setArg(0, C);
    calculateTreeLengthKernel.setArg(1, S);

    ocl.queue.enqueueNDRangeKernel(
            calculateTreeLengthKernel,
            NullRange,
            NDRange(sum),
            NullRange
    );
    Image3D centerlines= Image3D(
            ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_R, CL_SIGNED_INT8),
            size.x, size.y, size.z
        );

    if(getParamStr(parameters, "centerline-vtk-file") != "off" ||
    		getParamBool(parameters, "loop-removal")) {
    	// Do rasterization of centerline on CPU
    	// Transfer edges (size: sum2) and vertices (size: sum) buffers to host
    	int * verticesArray = new int[sum*3];
    	int * edgesArray = new int[sum2*2];
    	int * CArray = new int[sum];
    	int * SArray = new int[sum];

    	ocl.queue.enqueueReadBuffer(vertices, CL_FALSE, 0, sum*3*sizeof(int), verticesArray);
    	ocl.queue.enqueueReadBuffer(edges, CL_FALSE, 0, sum2*2*sizeof(int), edgesArray);
    	ocl.queue.enqueueReadBuffer(C, CL_FALSE, 0, sum*sizeof(int), CArray);
    	ocl.queue.enqueueReadBuffer(S, CL_FALSE, 0, sum*sizeof(int), SArray);

    	ocl.queue.finish();
    	float * radiusB = new float[totalSize];
    	ocl.queue.enqueueReadImage(radius, CL_FALSE, offset, region, 0, 0, radiusB);
    	std::vector<int3> vertices;
    	int counter = 0;
    	int * indexes = new int[sum];
    	for(int i = 0; i < sum; i++) {
    		if(SArray[CArray[i]] >= minTreeLength) {
				int3 v(verticesArray[i*3],verticesArray[i*3+1],verticesArray[i*3+2]);
				vertices.push_back(v);
				indexes[i] = counter;
				counter++;
    		}
    	}
    	std::vector<SIPL::int2> edges;
		int maxEdgeDistance = getParam(parameters, "max-edge-distance");
    	for(int i = 0; i < sum2; i++) {
    		if(SArray[CArray[edgesArray[i*2]]] >= minTreeLength && SArray[CArray[edgesArray[i*2+1]]] >= minTreeLength ) {
    			// Check length of edge
    			int3 A = vertices[indexes[edgesArray[i*2]]];
    			int3 B = vertices[indexes[edgesArray[i*2+1]]];
    			float distance = A.distance(B);
    			if(getParamStr(parameters, "centerline-vtk-file") != "off" &&
    					distance > maxEdgeDistance) {
					float3 direction(B.x-A.x,B.y-A.y,B.z-A.z);
					float3 Af(A.x,A.y,A.z);
					int previous = indexes[edgesArray[i*2]];
    				for(int j = maxEdgeDistance; j < distance; j += maxEdgeDistance) {
    					float3 newPos = Af + ((float)j/distance)*direction;
    					int3 newVertex(round(newPos.x), round(newPos.y), round(newPos.z));
    					// Create new vertex
    					vertices.push_back(newVertex);
    					// Add new edge
    					SIPL::int2 edge(previous, counter);
    					edges.push_back(edge);
    					previous = counter;
    					counter++;
    				}
    				// Connect previous vertex to B
    				SIPL::int2 edge(previous, indexes[edgesArray[i*2+1]]);
    				edges.push_back(edge);
    			} else {
					SIPL::int2 v(indexes[edgesArray[i*2]],indexes[edgesArray[i*2+1]]);
					edges.push_back(v);
    			}
    		}
    	}

    	// Remove loops from graph
    	removeLoops(vertices, edges, size);

    	ocl.queue.finish();
    	char * centerlinesData = createCenterlineVoxels(vertices, edges, radiusB, size);
    	ocl.queue.enqueueWriteImage(
    			centerlines,
    			CL_FALSE,
    			offset,
    			region,
    			0, 0,
    			centerlinesData
		);
		ocl.queue.enqueueWriteImage(
    			radius,
    			CL_FALSE,
    			offset,
    			region,
    			0, 0,
    			radiusB
		);

		if(getParamStr(parameters, "centerline-vtk-file") != "off")
			writeToVtkFile(parameters, vertices, edges);

    	delete[] verticesArray;
    	delete[] edgesArray;
    	delete[] CArray;
    	delete[] SArray;
    	delete[] indexes;
    } else {
    	// Do rasterization of centerline on GPU
    	Kernel RSTKernel(ocl.program, "removeSmallTrees");
		RSTKernel.setArg(0, edges);
		RSTKernel.setArg(1, vertices);
		RSTKernel.setArg(2, C);
		RSTKernel.setArg(3, S);
		RSTKernel.setArg(4, minTreeLength);
		if(no3Dwrite) {
			Buffer centerlinesBuffer = Buffer(
					ocl.context,
					CL_MEM_WRITE_ONLY,
					sizeof(char)*totalSize
			);

			initCharBuffer.setArg(0, centerlinesBuffer);
			ocl.queue.enqueueNDRangeKernel(
					initCharBuffer,
					NullRange,
					NDRange(totalSize),
					NullRange
			);

			RSTKernel.setArg(5, centerlinesBuffer);
			RSTKernel.setArg(6, size.x);
			RSTKernel.setArg(7, size.y);

			ocl.queue.enqueueNDRangeKernel(
					RSTKernel,
					NullRange,
					NDRange(sum2),
					NullRange
			);

			ocl.queue.enqueueCopyBufferToImage(
					centerlinesBuffer,
					centerlines,
					0,
					offset,
					region
			);

		} else {

			Kernel init3DImage(ocl.program, "init3DImage");
			init3DImage.setArg(0, centerlines);
			ocl.queue.enqueueNDRangeKernel(
				init3DImage,
				NullRange,
				NDRange(size.x, size.y, size.z),
				NullRange
			);

			RSTKernel.setArg(5, centerlines);

			ocl.queue.enqueueNDRangeKernel(
					RSTKernel,
					NullRange,
					NDRange(sum2),
					NullRange
			);
		}
    }

if(getParamBool(parameters, "timing")) {
    ocl.queue.enqueueMarker(&endEvent);
    ocl.queue.finish();
    startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    endEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &end);
    std::cout << "RUNTIME of removing small trees: " << (end-start)*1.0e-6 << " ms" << std::endl;
}
    return centerlines;
}
