#include "ridgeTraversalCenterlineExtraction.hpp"
#include <queue>
#include <vector>
#include <list>
#include "eigenanalysisOfHessian.hpp"
#include "timing.hpp"
#ifdef CPP11
#include <unordered_set>
using std::unordered_set;
#else
#include <boost/unordered_set.hpp>
using boost::unordered_set;
#endif

typedef struct point {
    float value;
    int x,y,z;
} point;

class PointComparison {
    public:
    bool operator() (const point &lhs, const point &rhs) const {
        return (lhs.value < rhs.value);
    }
};

float sign(float a) {
    return a < 0 ? -1.0f: 1.0f;
}

#define LPOS(a,b,c) (a)+(b)*(size.x)+(c)*(size.x*size.y)
#define POS(pos) pos.x+pos.y*size.x+pos.z*size.x*size.y
#define M(a,b,c) 1-sqrt(pow(T.Fx[a+b*size.x+c*size.x*size.y],2.0f) + pow(T.Fy[a+b*size.x+c*size.x*size.y],2.0f) + pow(T.Fz[a+b*size.x+c*size.x*size.y],2.0f))
#define SQR_MAG(pos) sqrt(pow(T.Fx[pos.x+pos.y*size.x+pos.z*size.x*size.y],2.0f) + pow(T.Fy[pos.x+pos.y*size.x+pos.z*size.x*size.y],2.0f) + pow(T.Fz[pos.x+pos.y*size.x+pos.z*size.x*size.y],2.0f))
#define SQR_MAG_SMALL(pos) sqrt(pow(T.FxSmall[pos.x+pos.y*size.x+pos.z*size.x*size.y],2.0f) + pow(T.FySmall[pos.x+pos.y*size.x+pos.z*size.x*size.y],2.0f) + pow(T.FzSmall[pos.x+pos.y*size.x+pos.z*size.x*size.y],2.0f))

char * runRidgeTraversal(TubeSegmentation &T, SIPL::int3 size, SIPL::float3 spacing, paramList &parameters, std::stack<CenterlinePoint> centerlineStack) {

    float Thigh = getParam(parameters, "tdf-high"); // 0.6
    int Dmin = getParam(parameters, "min-distance");
    float Mlow = getParam(parameters, "m-low"); // 0.2
    float Tlow = getParam(parameters, "tdf-low"); // 0.4
    int maxBelowTlow = getParam(parameters, "max-below-tdf-low"); // 2
    float minMeanTube = getParam(parameters, "min-mean-tdf"); //0.6
    int TreeMin = getParam(parameters, "min-tree-length"); // 200
    const int totalSize = size.x*size.y*size.z;

    int * centerlines = new int[totalSize]();
    INIT_TIMER

    // Create queue
    std::priority_queue<point, std::vector<point>, PointComparison> queue;

    START_TIMER
    // Collect all valid start points
    #pragma omp parallel for
    for(int z = 2; z < size.z-2; z++) {
        for(int y = 2; y < size.y-2; y++) {
            for(int x = 2; x < size.x-2; x++) {
                if(T.TDF[LPOS(x,y,z)] < Thigh)
                    continue;

                int3 pos(x,y,z);
                bool valid = true;
                for(int a = -1; a < 2; a++) {
                    for(int b = -1; b < 2; b++) {
                        for(int c = -1; c < 2; c++) {
                            int3 nPos(x+a,y+b,z+c);
                            if(SQR_MAG(nPos) < SQR_MAG(pos)) {
                                valid = false;
                                break;
                            }
                        }
                    }
                }

                if(valid) {
                    point p;
                    p.value = T.TDF[LPOS(x,y,z)];
                    p.x = x;
                    p.y = y;
                    p.z = z;
                    #pragma omp critical
                    queue.push(p);
                }
            }
        }
    }

    std::cout << "Processing " << queue.size() << " valid start points" << std::endl;
    if(queue.size() == 0) {
    	throw SIPL::SIPLException("no valid start points found", __LINE__, __FILE__);
    }
    STOP_TIMER("finding start points")
    START_TIMER
    int counter = 1;
    T.TDF[0] = 0;
    T.Fx[0] = 1;
    T.Fy[0] = 0;
    T.Fz[0] = 0;


    // Create a map of centerline distances
    unordered_map<int, int> centerlineDistances;

    // Create a map of centerline stacks
    unordered_map<int, std::stack<CenterlinePoint> > centerlineStacks;

    while(!queue.empty()) {
        // Traverse from new start point
        point p = queue.top();
        queue.pop();

        // Has it been handled before?
        if(centerlines[LPOS(p.x,p.y,p.z)] == 1)
            continue;

        unordered_set<int> newCenterlines;
        newCenterlines.insert(LPOS(p.x,p.y,p.z));
        int distance = 1;
        int connections = 0;
        int prevConnection = -1;
        int secondConnection = -1;
        float meanTube = T.TDF[LPOS(p.x,p.y,p.z)];

        // Create new stack for this centerline
        std::stack<CenterlinePoint> stack;
        CenterlinePoint startPoint;
        startPoint.pos.x = p.x;
        startPoint.pos.y = p.y;
        startPoint.pos.z = p.z;

        stack.push(startPoint);

        // For each direction
        for(int direction = -1; direction < 3; direction += 2) {
            int belowTlow = 0;
            int3 position(p.x,p.y,p.z);
            float3 t_i = getTubeDirection(T, position, size, spacing);
            t_i.x *= direction;
            t_i.y *= direction;
            t_i.z *= direction;
            float3 t_i_1;
            t_i_1.x = t_i.x;
            t_i_1.y = t_i.y;
            t_i_1.z = t_i.z;


            // Traverse
            while(true) {
                int3 maxPoint(0,0,0);

                // Check for out of bounds
                if(position.x < 3 || position.x > size.x-3 || position.y < 3 || position.y > size.y-3 || position.z < 3 || position.z > size.z-3)
                    break;

                // Try to find next point from all neighbors
                for(int a = -1; a < 2; a++) {
                    for(int b = -1; b < 2; b++) {
                        for(int c = -1; c < 2; c++) {
                            int3 n(position.x+a,position.y+b,position.z+c);
                            if((a == 0 && b == 0 && c == 0) || T.TDF[POS(n)] == 0.0f)
                                continue;

                            float3 dir((float)(n.x-position.x),(float)(n.y-position.y),(float)(n.z-position.z));
                            dir = dir.normalize();
                            if( (dir.x*t_i.x+dir.y*t_i.y+dir.z*t_i.z) <= 0.1)
                                continue;

                            if(T.radius[POS(n)] >= 1.5f) {
                                if(M(n.x,n.y,n.z) > M(maxPoint.x,maxPoint.y,maxPoint.z))
                                maxPoint = n;
                            } else {
                                if(T.TDF[LPOS(n.x,n.y,n.z)]*M(n.x,n.y,n.z) > T.TDF[POS(maxPoint)]*M(maxPoint.x,maxPoint.y,maxPoint.z))
                                maxPoint = n;
                            }

                        }
                    }
                }

                if(maxPoint.x+maxPoint.y+maxPoint.z > 0) {
                    // New maxpoint found, check it!
                    if(centerlines[LPOS(maxPoint.x,maxPoint.y,maxPoint.z)] > 0) {
                        // Hit an existing centerline
                        if(prevConnection == -1) {
                            prevConnection = centerlines[LPOS(maxPoint.x,maxPoint.y,maxPoint.z)];
                        } else {
                            if(prevConnection ==centerlines[LPOS(maxPoint.x,maxPoint.y,maxPoint.z)]) {
                                // A loop has occured, reject this centerline
                                connections = 5;
                            } else {
                                secondConnection = centerlines[LPOS(maxPoint.x,maxPoint.y,maxPoint.z)];
                            }
                        }
                        break;
                    } else if(M(maxPoint.x,maxPoint.y,maxPoint.z) < Mlow || (belowTlow > maxBelowTlow && T.TDF[LPOS(maxPoint.x,maxPoint.y,maxPoint.z)] < Tlow)) {
                        // New point is below thresholds
                        break;
                    } else if(newCenterlines.count(LPOS(maxPoint.x,maxPoint.y,maxPoint.z)) > 0) {
                        // Loop detected!
                        break;
                    } else {
                        // Point is OK, proceed to add it and continue
                        if(T.TDF[LPOS(maxPoint.x,maxPoint.y,maxPoint.z)] < Tlow) {
                            belowTlow++;
                        } else {
                            belowTlow = 0;
                        }

                        // Update direction
                        //float3 e1 = getTubeDirection(T, maxPoint,size.x,size.y,size.z);

                        //TODO: check if all eigenvalues are negative, if so find the egeinvector that best matches
                        float3 lambda, e1, e2, e3;
                        doEigen(T, maxPoint, size, spacing, &lambda, &e1, &e2, &e3);
                        if((lambda.x < 0 && lambda.y < 0 && lambda.z < 0)) {
                            if(fabs(t_i.dot(e3)) > fabs(t_i.dot(e2))) {
                                if(fabs(t_i.dot(e3)) > fabs(t_i.dot(e1))) {
                                    e1 = e3;
                                }
                            } else if(fabs(t_i.dot(e2)) > fabs(t_i.dot(e1))) {
                                e1 = e2;
                            }
                        }


                        float maintain_dir = sign(e1.dot(t_i));
                        float3 vec_sum;
                        vec_sum.x = maintain_dir*e1.x + t_i.x + t_i_1.x;
                        vec_sum.y = maintain_dir*e1.y + t_i.y + t_i_1.y;
                        vec_sum.z = maintain_dir*e1.z + t_i.z + t_i_1.z;
                        vec_sum = vec_sum.normalize();
                        t_i_1 = t_i;
                        t_i = vec_sum;

                        // update position
                        position = maxPoint;
                        distance ++;
                        newCenterlines.insert(LPOS(maxPoint.x,maxPoint.y,maxPoint.z));
                        meanTube += T.TDF[LPOS(maxPoint.x,maxPoint.y,maxPoint.z)];

                        // Create centerline point
                        CenterlinePoint p;
                        p.pos = position;
                        p.next = &(stack.top()); // add previous
                        if(T.radius[POS(p.pos)] > 3.0f) {
                            p.large = true;
                        } else {
                            p.large = false;
                        }

                        // Add point to stack
                        stack.push(p);
                    }
                } else {
                    // No maxpoint found, stop!
                    break;
                }

            } // End traversal
        } // End for each direction

        // Check to see if new traversal can be added
        //std::cout << "Finished. Distance " << distance << " meanTube: " << meanTube/distance << std::endl;
        if(distance > Dmin && meanTube/distance > minMeanTube && connections < 2) {
            //std::cout << "Finished. Distance " << distance << " meanTube: " << meanTube/distance << std::endl;
            //std::cout << "------------------- New centerlines added #" << counter << " -------------------------" << std::endl;

            unordered_set<int>::iterator usit;
            if(prevConnection == -1) {
                // No connections
                for(usit = newCenterlines.begin(); usit != newCenterlines.end(); usit++) {
                    centerlines[*usit] = counter;
                }
                centerlineDistances[counter] = distance;
                centerlineStacks[counter] = stack;
                counter ++;
            } else {
                // The first connection

                std::stack<CenterlinePoint> prevConnectionStack = centerlineStacks[prevConnection];
                while(!stack.empty()) {
                    prevConnectionStack.push(stack.top());
                    stack.pop();
                }

                for(usit = newCenterlines.begin(); usit != newCenterlines.end(); usit++) {
                    centerlines[*usit] = prevConnection;
                }
                centerlineDistances[prevConnection] += distance;
                if(secondConnection != -1) {
                    // Two connections, move secondConnection to prevConnection
                    std::stack<CenterlinePoint> secondConnectionStack = centerlineStacks[secondConnection];
                    centerlineStacks.erase(secondConnection);
                    while(!secondConnectionStack.empty()) {
                        prevConnectionStack.push(secondConnectionStack.top());
                        secondConnectionStack.pop();
                    }

                    #pragma omp parallel for
                    for(int i = 0; i < totalSize;i++) {
                        if(centerlines[i] == secondConnection)
                            centerlines[i] = prevConnection;
                    }
                    centerlineDistances[prevConnection] += centerlineDistances[secondConnection];
                    centerlineDistances.erase(secondConnection);
                }

                centerlineStacks[prevConnection] = prevConnectionStack;
            }
        } // end if new point can be added
    } // End while queue is not empty
    std::cout << "Finished traversal" << std::endl;
    STOP_TIMER("traversal")
    START_TIMER

    if(centerlineDistances.size() == 0) {
        //throw SIPL::SIPLException("no centerlines were extracted");
        char * returnCenterlines = new char[totalSize]();
        return returnCenterlines;
    }

    // Find largest connected tree and all trees above a certain size
    unordered_map<int, int>::iterator it;
    int max = centerlineDistances.begin()->first;
    std::list<int> trees;
    for(it = centerlineDistances.begin(); it != centerlineDistances.end(); it++) {
        if(it->second > centerlineDistances[max])
            max = it->first;
        if(it->second > TreeMin)
            trees.push_back(it->first);
    }
    std::list<int>::iterator it2;
    // TODO: if use the method with TreeMin have to add them to centerlineStack also
    centerlineStack = centerlineStacks[max];
    for(it2 = trees.begin(); it2 != trees.end(); it2++) {
        while(!centerlineStacks[*it2].empty()) {
            centerlineStack.push(centerlineStacks[*it2].top());
            centerlineStacks[*it2].pop();
        }
    }

    char * returnCenterlines = new char[totalSize]();
    // Mark largest tree with 1, and rest with 0
    #pragma omp parallel for
    for(int i = 0; i < totalSize;i++) {
        if(centerlines[i] == max) {
        //if(centerlines[i] > 0) {
            returnCenterlines[i] = 1;
        } else {
            bool valid = false;
            for(it2 = trees.begin(); it2 != trees.end(); it2++) {
                if(centerlines[i] == *it2) {
                    returnCenterlines[i] = 1;
                    valid = true;
                    break;
                }
            }
            if(!valid)
                returnCenterlines[i] = 0;

        }
    }
    STOP_TIMER("finding largest tree")

	delete[] centerlines;
    return returnCenterlines;
}
