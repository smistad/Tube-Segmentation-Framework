#include "globalCenterlineExtraction.hpp"
#include "eigenanalysisOfHessian.hpp"
#include "SIPL/Types.hpp"
#ifdef CPP11
#include <unordered_set>
using std::unordered_set;
#else
#include <boost/unordered_set.hpp>
using boost::unordered_set;
#endif
#include <stack>
#include <queue>
#include <algorithm>

using namespace SIPL;

#define MAX(a,b) a > b ? a : b

#define LPOS(a,b,c) (a)+(b)*(size.x)+(c)*(size.x*size.y)
#define POS(pos) pos.x+pos.y*size.x+pos.z*size.x*size.y
#define M(a,b,c) 1-sqrt(pow(T.Fx[a+b*size.x+c*size.x*size.y],2.0f) + pow(T.Fy[a+b*size.x+c*size.x*size.y],2.0f) + pow(T.Fz[a+b*size.x+c*size.x*size.y],2.0f))
#define SQR_MAG(pos) sqrt(pow(T.Fx[pos.x+pos.y*size.x+pos.z*size.x*size.y],2.0f) + pow(T.Fy[pos.x+pos.y*size.x+pos.z*size.x*size.y],2.0f) + pow(T.Fz[pos.x+pos.y*size.x+pos.z*size.x*size.y],2.0f))
#define SQR_MAG_SMALL(pos) sqrt(pow(T.FxSmall[pos.x+pos.y*size.x+pos.z*size.x*size.y],2.0f) + pow(T.FySmall[pos.x+pos.y*size.x+pos.z*size.x*size.y],2.0f) + pow(T.FzSmall[pos.x+pos.y*size.x+pos.z*size.x*size.y],2.0f))

class CrossSectionCompare {
private:
	float * dist;
public:
	CrossSectionCompare(float * dist) { this->dist = dist; };
	bool operator() (const CrossSection * a, const CrossSection * b) {
		return dist[a->index] > dist[b->index];
	};
};

std::vector<CrossSection *> createGraph(TubeSegmentation &T, SIPL::int3 size) {
	// Create vector
	std::vector<CrossSection *> sections;
	float threshold = 0.5f;

	// Go through TS.TDF and add all with TDF above threshold
	int counter = 0;
	float thetaLimit = 0.5;
	for(int z = 1; z < size.z-1; z++) {
	for(int y = 1; y < size.y-1; y++) {
	for(int x = 1; x < size.x-1; x++) {
		int3 pos(x,y,z);
		float tdf = T.TDF[POS(pos)];
		if(tdf > threshold) {
			int maxD = std::min(std::max((int)round(T.radius[POS(pos)]), 1), 5);
		    //std::cout << SQR_MAG(pos) << " " << SQR_MAG_SMALL(pos) << std::endl;
			//std::cout << "radius" << TS.radius[POS(pos)] << std::endl;
			//std::cout << "maxD "<< maxD <<std::endl;
			float3 e1 = getTubeDirection(T, pos, size);
			bool invalid = false;
		    for(int a = -maxD; a <= maxD; a++) {
		    for(int b = -maxD; b <= maxD; b++) {
		    for(int c = -maxD; c <= maxD; c++) {
		        if(a == 0 && b == 0 && c == 0)
		            continue;
		        const int3 n = pos + int3(a,b,c);
		        if(!inBounds(n, size))
		        	continue;
		        float3 r(a,b,c);
		        const float dp = e1.dot(r);
		        float3 r_projected = float3(r.x-e1.x*dp,r.y-e1.y*dp,r.z-e1.z*dp);
		        float3 rn = r.normalize();
		        float3 rpn = r_projected.normalize();
		        float theta = acos((double)rn.dot(rpn));
		        //std::cout << "theta: " << theta << std::endl;

		        if((theta < thetaLimit && r.length() < maxD-0.5f)) {
		        	//std::cout << SQR_MAG(n) << std::endl;
		            /*
		        	if(T.radius[POS(pos)]<= 3) {
		            //if(TS.TDF[POS(n)] > TS.TDF[POS(pos)]) {
		            if(SQR_MAG_SMALL(n) < SQR_MAG_SMALL(pos)) {
		                invalid = true;
		                break;
		            }
		            } else {
		            */
		            if(SQR_MAG(n) < SQR_MAG(pos)) {
		            //if(TS.TDF[POS(n)] > TS.TDF[POS(pos)]) {
		                invalid = true;
		                break;
		            }
		            //}
		        }
		    }}}
		    if(!invalid) {
				CrossSection * cs = new CrossSection;
				cs->pos = pos;
				cs->TDF = tdf;
				cs->label = -1;//counter;
				cs->direction = e1;
				counter++;
				sections.push_back(cs);
		    }
		}
	}}}


	// For each cross section c_i
	for(int i = 0; i < sections.size(); i++) {
		CrossSection * c_i = sections[i];
		// For each cross section c_j
		for(int j = 0; j < i; j++) {
			CrossSection * c_j = sections[j];
			// If all criterias are ok: Add c_j as neighbor to c_i
			if(c_i->pos.distance(c_j->pos) < 4 && !(c_i->pos == c_j->pos)) {
				float3 e1_i = c_i->direction;
				float3 e1_j = c_j->direction;
				int3 cint = c_i->pos - c_j->pos;
				float3 c = cint.normalize();

				if(acos((double)fabs(e1_i.dot(e1_j))) > 1.05) // 60 degrees
					continue;

				if(acos((double)fabs(e1_i.dot(c))) > 1.05)
					continue;

				if(acos((double)fabs(e1_j.dot(c))) > 1.05)
					continue;

				int distance = ceil(c_i->pos.distance(c_j->pos));
				float3 direction(c_j->pos.x-c_i->pos.x,c_j->pos.y-c_i->pos.y,c_j->pos.z-c_i->pos.z);
				bool invalid = false;
				for(int i = 0; i < distance; i++) {
					float frac = (float)i/distance;
					float3 n = c_i->pos + frac*direction;
					int3 in(round(n.x),round(n.y),round(n.z));
					//float3 e1 = getTubeDirection(TS, in, size);
					//cost += (1-fabs(a->direction.dot(e1)))+(1-fabs(b->direction.dot(e1)));
					/*
					if(T.intensity[POS(in)] > 0.5f) {
						invalid = true;
						break;
					}
					*/
				}
				//if(invalid)
				//	continue;


				c_i->neighbors.push_back(c_j);
				c_j->neighbors.push_back(c_i);
				//sectionPairs.push_back(c_i);
			}
			// If no pair is found, dont add it
		}
	}

	std::vector<CrossSection *> sectionPairs;
	for(int i = 0; i < sections.size(); i++) {
		CrossSection * c_i = sections[i];
		if(c_i->neighbors.size()>0) {
			sectionPairs.push_back(c_i);
		}
	}

	return sectionPairs;
}


bool segmentCompare(Segment * a, Segment * b) {
	return a->benefit > b->benefit;
}

bool segmentInSegmentation(Segment * s, unordered_set<int> &segmentation, int3 size) {
	bool in = false;
	for(int i = 0; i < s->sections.size(); i++) {
		CrossSection * c = s->sections[i];
		if(segmentation.find(POS(c->pos)) != segmentation.end()) {
			in = true;
			break;
		}
	}
	return in;
}

float calculateBenefit(CrossSection * a, CrossSection * b, TubeSegmentation &TS, int3 size) {
	float benefit = 0.0f;
	int distance = ceil(a->pos.distance(b->pos));
	float3 direction(b->pos.x-a->pos.x,b->pos.y-a->pos.y,b->pos.z-a->pos.z);
	for(int i = 0; i < distance; i++) {
		float frac = (float)i/distance;
		float3 n = a->pos + frac*direction;
		int3 in(round(n.x),round(n.y),round(n.z));
		benefit += (TS.TDF[POS(in)]);
	}

	return benefit;
}

void inverseGradientRegionGrowing(Segment * s, TubeSegmentation &TS, unordered_set<int> &segmentation, int3 size) {
    std::vector<int3> centerpoints;
	for(int c = 0; c < s->sections.size()-1; c++) {
		CrossSection * a = s->sections[c];
		CrossSection * b = s->sections[c+1];
		int distance = ceil(a->pos.distance(b->pos));
		float3 direction(b->pos.x-a->pos.x,b->pos.y-a->pos.y,b->pos.z-a->pos.z);
		for(int i = 0; i < distance; i++) {
			float frac = (float)i/distance;
			float3 n = a->pos + frac*direction;
			int3 in(round(n.x),round(n.y),round(n.z));
			centerpoints.push_back(in);
		}
		segmentation.insert(POS(a->pos));//test
		segmentation.insert(POS(b->pos));//test
	}

	// Dilate the centerline
	std::vector<int3> dilatedCenterline;
	for(int i = 0; i < centerpoints.size(); i++) {
		int3 pos = centerpoints[i];
		for(int a = -1; a < 2; a++) {
		for(int b = -1; b < 2; b++) {
		for(int c = -1; c < 2; c++) {
			int3 n = pos + int3(a,b,c);
			if(inBounds(n, size) && segmentation.find(POS(n)) == segmentation.end()) {
				segmentation.insert(POS(n));
				dilatedCenterline.push_back(n);
			}
		}}}
	}
	/*

	std::queue<int3> queue;
	for(int3 pos : dilatedCenterline) {
		for(int a = -1; a < 2; a++) {
		for(int b = -1; b < 2; b++) {
		for(int c = -1; c < 2; c++) {
			int3 n = pos + int3(a,b,c);
			if(inBounds(n, size) && segmentation.find(POS(n)) == segmentation.end()) {
				queue.push(n);
			}
		}}}
	}

	while(!queue.empty()) {
		int3 X = queue.front();
		float FNXw = SQR_MAG(X);
		queue.pop();
		for(int a = -1; a < 2; a++) {
		for(int b = -1; b < 2; b++) {
		for(int c = -1; c < 2; c++) {
			if(a == 0 && b == 0 && c == 0)
				continue;

			int3 Y = X + int3(a,b,c);
			if(inBounds(Y, size) && segmentation.find(POS(Y)) == segmentation.end()) {

				float3 FNY;
				FNY.x = TS.Fx[POS(Y)];
				FNY.y = TS.Fy[POS(Y)];
				FNY.z = TS.Fz[POS(Y)];
				float FNYw = FNY.length();
				FNY = FNY.normalize();
				if(FNYw > FNXw || FNXw < 0.1f) {

					int3 Z;
					float maxDotProduct = -2.0f;
					for(int a2 = -1; a2 < 2; a2++) {
					for(int b2 = -1; b2 < 2; b2++) {
					for(int c2 = -1; c2 < 2; c2++) {
						if(a2 == 0 && b2 == 0 && c2 == 0)
							continue;
						int3 Zc;
						Zc.x = Y.x+a2;
						Zc.y = Y.y+b2;
						Zc.z = Y.z+c2;
						float3 YZ;
						YZ.x = Zc.x-Y.x;
						YZ.y = Zc.y-Y.y;
						YZ.z = Zc.z-Y.z;
						YZ = YZ.normalize();
						if(FNY.dot(YZ) > maxDotProduct) {
							maxDotProduct = FNY.dot(YZ);
							Z = Zc;
						}
					}}}

					if(Z.x == X.x && Z.y == X.y && Z.z == X.z) {
						segmentation.insert(POS(X));
						queue.push(Y);
					}
				}
			}
		}}}
	}
	*/

}

std::vector<Segment *> createSegments(OpenCL &ocl, TubeSegmentation &TS, std::vector<CrossSection *> &crossSections, SIPL::int3 size) {
	// Create segment vector
	std::vector<Segment *> segments;

	// Do a graph component labeling
	unordered_set<int> visited;
    int labelCounter = 0;
    std::vector<std::vector<CrossSection *> > labels;
	for(int i = 0; i < crossSections.size(); i++) {
		CrossSection * c = crossSections[i];
		// Do a bfs on c
		// Check to see if point has been processed before doing a BFS
		if(visited.find(c->label) != visited.end())
			continue;

        c->label = labelCounter;
        labelCounter++;
        std::vector<CrossSection *> list;

		std::stack<CrossSection *> stack;
		stack.push(c);
		while(!stack.empty()) {
			CrossSection * current = stack.top();
			stack.pop();
			// Check label of neighbors to see if they have been added
			if(current->label != c->label || c->pos == current->pos) {
                list.push_back(current);
				// Change label of neighbors if not
				current->label = c->label;
				// Add neighbors to stack

				for(int j = 0; j < current->neighbors.size(); j++) {
					CrossSection * n = current->neighbors[j];
					if(n->label != c->label)
						stack.push(n);
				}
			}
		}
		visited.insert(c->label);
        labels.push_back(list);
	}

	std::cout << "finished graph component labeling" << std::endl;

	// Do a floyd warshall all pairs shortest path
	int totalSize = crossSections.size();
	std::cout << "number of cross sections is " << totalSize << std::endl;

    // For each label
	for(int i = 0; i < labels.size(); i++) {
		std::vector<CrossSection *> list = labels[i];
        // Do floyd warshall on all pairs
        int totalSize = list.size();
        float * dist = new float[totalSize*totalSize];
        int * pred = new int[totalSize*totalSize];

        for(int u = 0; u < totalSize; u++) {
            CrossSection * U = list[u];
            U->index = u;
        }
        #define DPOS(U, V) V+U*totalSize
        // For each cross section U
        for(int u = 0; u < totalSize; u++) {
            CrossSection * U = list[u];
            // For each cross section V
            for(int v = 0; v < totalSize; v++) {
                dist[DPOS(u,v)] = 99999999;
                pred[DPOS(u,v)] = -1;
            }
            dist[DPOS(U->index,U->index)] = 0;
            for(int j = 0; j < U->neighbors.size(); j++) {
                CrossSection * V = U->neighbors[j];
                // TODO calculate more advanced weight
                dist[DPOS(U->index,V->index)] = ceil(U->pos.distance(V->pos)) - calculateBenefit(U, V, TS, size); //(1-V->TDF);
                pred[DPOS(U->index,V->index)] = U->index;
            }
        }
        for(int t = 0; t < totalSize; t++) {
            //CrossSection * T = crossSections[t];
            //std::cout << "processing t=" << t << std::endl;
            // For each cross section U
            for(int u = 0; u < totalSize; u++) {
                //CrossSection * U = crossSections[u];
                // For each cross section V
                for(int v = 0; v < totalSize; v++) {
                    //CrossSection * V = crossSections[v];
                    float newLength = dist[DPOS(u, t)] + dist[DPOS(t,v)];
                    if(newLength < dist[DPOS(u,v)]) {
                        dist[DPOS(u,v)] = newLength;
                        pred[DPOS(u,v)] = pred[DPOS(t,v)];
                    }
                }
            }
        }

        for(int s = 0; s < list.size(); s++) { // Source
            CrossSection * S = list[s];
            for(int t = 0; t < list.size(); t++) { // Target
                CrossSection * T = list[t];
                if(S->label == T->label && S->index != T->index) {
                    Segment * segment = new Segment;
                    // add all cross sections in segment
                    float benefit = 0.0f;
                    segment->sections.push_back(T);
                    int current = T->index;
                    while(current != S->index) {
                        CrossSection * C = list[current];
                        segment->sections.push_back(C);
                        current = pred[DPOS(S->index,current)];// get predecessor
                        benefit += calculateBenefit(C, list[current], TS, size);
                    }
                    segment->sections.push_back(list[current]);
                    segment->benefit = benefit;
                    segments.push_back(segment);
                }
            }
        }

        delete[] dist;
        delete[] pred;
    }
	std::cout << "finished performing floyd warshall" << std::endl;

	std::cout << "finished creating segments" << std::endl;
	std::cout << "total number of segments is " << segments.size() << std::endl;


	// Sort the segment vector on benefit
	std::sort(segments.begin(), segments.end(), segmentCompare);
	unordered_set<int> segmentation;

	// Go through sorted vector and do a region growing
	std::vector<Segment *> filteredSegments;
	int counter = 0;
	for(int i = 0; i < segments.size(); i++) {
		Segment * s = segments[i];
		if(!segmentInSegmentation(s, segmentation, size)) {
			//std::cout << "adding segment with benefit: " << s->benefit << std::endl;
			// Do region growing and Add all segmented voxels to a set
			inverseGradientRegionGrowing(s, TS, segmentation, size);
			filteredSegments.push_back(s);
			s->index = counter;
			counter++;
		}
	}

	std::cout << "total number of segments after remove overlapping segments " << filteredSegments.size() << std::endl;

	return filteredSegments;
}

int selectRoot(std::vector<Segment *> segments, float minBenefit) {
	int root = -1;
	for(int i = 0; i < segments.size(); i++) {
	    if(segments[i]->benefit > minBenefit) {
	        if(root == -1) {
	            root = i;
	        } else if(segments[i]->benefit > segments[root]->benefit) {
                root = i;
	        }
	    }
	}
	return root;
}

void DFS(Segment * current, int * ordering, int &counter, unordered_set<int> &visited) {
	if(visited.find(current->index) != visited.end())
		return;
	ordering[counter] = current->index;
	//std::cout << counter << ": " << current->index << std::endl;
	counter++;
	for(int i = 0; i < current->connections.size(); i++) {
		Connection * edge = current->connections[i];
		DFS(edge->target, ordering, counter, visited);
	}
	visited.insert(current->index);
}

int * createDepthFirstOrdering(std::vector<Segment *> segments, int root, int &Ns) {
	int * ordering = new int[segments.size()];
	int counter = 0;

	// Give imdexes to segments
	for(int i = 0; i < segments.size(); i++) {
		segments[i]->index = i;
	}

	unordered_set<int> visited;

	DFS(segments[root], ordering, counter, visited);

	Ns = counter;
	int * reversedOrdering = new int[Ns];
	for(int i = 0; i < Ns; i++) {
		reversedOrdering[i] = ordering[Ns-i-1];
	}

	delete[] ordering;
	return reversedOrdering;
}

class ConnectionComparator {
public:
	bool operator()(Connection * a, Connection *b) const {
		return a->cost > b->cost;
	}
};

std::vector<Segment *> minimumSpanningTree(Segment * root, int3 size) {
	// Need a priority queue on Connection objects based on the cost
	std::priority_queue<Connection *, std::vector<Connection *>, ConnectionComparator> queue;
	std::vector<Segment *> result;
	unordered_set<int> visited;
	result.push_back(root);
	visited.insert(root->index);

	// Add all connections of the root to the queue
	for(int i = 0; i < root->connections.size(); i++) {
		Connection * c = root->connections[i];
		queue.push(c);
	}
	// Remove connections from root
	root->connections = std::vector<Connection *>();

	while(!queue.empty()) {
	// Select minimum connection
	// Check if target is already added
	// if not, add all of its connection to the queue
	// add this connection to the source
	// Add target segment and clear its connections
	// Also add cost to the segment object
		Connection * c = queue.top();
		//std::cout << c->cost << std::endl;
		queue.pop();
		if(visited.find(c->target->index) != visited.end())
			continue;

		for(int i = 0; i < c->target->connections.size(); i++) {
			Connection * cn = c->target->connections[i];
			if(visited.find(cn->target->index) == visited.end())
				queue.push(cn);
		}

		c->source->connections.push_back(c);
		// c->target->connections.clear(); doest his delete the objects?
		c->target->connections = std::vector<Connection *>();
		c->target->cost = c->cost;
		result.push_back(c->target);
		visited.insert(c->target->index);
	}

	return result;
}

std::vector<Segment *> findOptimalSubtree(std::vector<Segment *> segments, int * depthFirstOrdering, int Ns) {

	float * score = new float[Ns]();
	float r = 3.0;

	// Stage 1 bottom up
	for(int j = 0; j < Ns; j++) {
		int mj = depthFirstOrdering[j];
		score[mj] = segments[mj]->benefit - r * segments[mj]->cost;
		/*std::cout << "cross sections: " << segments[mj]->sections.size() << " benefit: "
				<< segments[mj]->benefit << " cost: " << segments[mj]->cost <<
				" children: " << segments[mj]->connections.size() << std::endl;*/
		// For all children of mj
		for(int n = 0; n < segments[mj]->connections.size(); n++) {
			Connection * c = segments[mj]->connections[n];
			int k = c->target->index; //child
			if(score[k] >= 0)
				score[mj] += score[k];
		}
	}

	// Stage 2 top down
	bool * v = new bool[Ns];
	for(int i = 1; i < Ns; i++)
		v[i] = false;
	v[0] = true;

	for(int j = Ns-1; j >= 0; j--) {
		int mj = depthFirstOrdering[j];
		if(v[mj]) {
			// For all children of mj
			for(int n = 0; n < segments[mj]->connections.size(); n++) {
				Connection * c = segments[mj]->connections[n];
				int k = c->target->index; //child
				if(score[k] >= 0)
					v[k] = true;
			}
		}
	}

	delete[] score;

	std::vector<Segment *> finalSegments;
	for(int i = 0; i < Ns; i++) {
		if(v[i]) {
			finalSegments.push_back(segments[i]);

			// for all children, check if they are true in v, if not remove connections
			std::vector<Connection *> connections;
			for(int n = 0; n < segments[i]->connections.size(); n++) {
				Connection * c = segments[i]->connections[n];
				int k = c->target->index; //child
				if(v[k]) {
					// keep connection
					connections.push_back(c);
				} else {
					delete c;
				}
			}
			segments[i]->connections = connections;
		}
	}
	delete[] v;
	return finalSegments;
}

float calculateConnectionCost(CrossSection * a, CrossSection * b, TubeSegmentation &TS, int3 size) {
	float cost = 0.0f;
	int distance = ceil(a->pos.distance(b->pos));
	float3 direction(b->pos.x-a->pos.x,b->pos.y-a->pos.y,b->pos.z-a->pos.z);
	float maxIntensity = -1.0f;
	for(int i = 0; i < distance; i++) {
		float frac = (float)i/distance;
		float3 n = a->pos + frac*direction;
		int3 in(round(n.x),round(n.y),round(n.z));
		cost += /*SQR_MAG(in) +*/ (1.0f-TS.TDF[POS(in)]);
        //float3 e1 = getTubeDirection(TS, in, size);
        //cost += (1-fabs(a->direction.dot(e1)))+(1-fabs(b->direction.dot(e1)));
		if(TS.intensity[POS(in)] > maxIntensity) {
			maxIntensity = TS.intensity[POS(in)];
		}
	}
	/*
	if(maxIntensity > 0.2 && maxIntensity < 0.3) {
		cost = cost*2;
	}
	if(maxIntensity >= 0.3 && maxIntensity < 0.5) {
		cost = cost*4;
	}
	if(maxIntensity >= 0.5) {
		cost = cost*8;
	}
	*/
	return cost;
}

void createConnections(TubeSegmentation &TS, std::vector<Segment *> segments, int3 size) {
	// For all pairs of segments
	for(int k = 0; k < segments.size(); k++) {
		Segment * s_k = segments[k];
		for(int l = 0; l < k; l++) {
			Segment * s_l = segments[l];
			// For each C_k, cross sections in S_k, calculate costs and select the one with least cost
			float bestCost = 999999999.0f;
			CrossSection * c_k_best, * c_l_best;
			bool found = false;
			for(int i = 0; i < s_k->sections.size(); i++){
				CrossSection * c_k = s_k->sections[i];
				for(int j = 0; j < s_l->sections.size(); j++){
					CrossSection * c_l = s_l->sections[j];
					if(c_k->pos.distance(c_l->pos) > 25)
						continue;

					float3 c(c_k->pos.x-c_l->pos.x, c_k->pos.y-c_l->pos.y,c_k->pos.z-c_l->pos.z);
					c = c.normalize();
					if(acos(fabs(c_k->direction.dot(c))) > 1.05f)
						continue;
					if(acos(fabs(c_l->direction.dot(c))) > 1.05f)
						continue;

					/*
                    float rk = TS.radius[POS(c_k->pos)];
                    float rl = TS.radius[POS(c_l->pos)];

                    if(rk > 2 || rl > 2) {
                        if(std::max(rk,rl) / std::min(rk,rl) >= 2)
                            continue;
                    }
                    */

					float cost = calculateConnectionCost(c_k, c_l, TS, size);
					if(cost < bestCost) {
						bestCost = cost;
						c_k_best = c_k;
						c_l_best = c_l;
						found = true;
					}
				}
			}


			// See if they are allowed to connect
			if(found) {
				/*if(bestCost < 2) {
					std::cout << bestCost << std::endl;
					std::cout << "labels: " << c_k_best->label << " " << c_l_best->label << std::endl;
					std::cout << "distance: " << c_k_best->pos.distance(c_l_best->pos) << std::endl;
				}*/
				// If so, create connection object and add to segemnt
				Connection * c = new Connection;
				c->cost = bestCost;
				c->source = s_k;
				c->source_section = c_k_best;
				c->target = s_l;
				c->target_section = c_l_best;
				s_k->connections.push_back(c);
				Connection * c2 = new Connection;
				c2->cost = bestCost;
				c2->source = s_l;
				c2->source_section = c_l_best;
				c2->target = s_k;
				c2->target_section = c_k_best;
				s_l->connections.push_back(c2);
			}
		}
	}
}

