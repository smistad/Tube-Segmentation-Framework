#include "inputOutput.hpp"
#include <fstream>
using namespace SIPL;

void writeToVtkFile(paramList &parameters, std::vector<int3> vertices, std::vector<SIPL::int2> edges) {
	// Write to file
	std::ofstream file;
	file.open(getParamStr(parameters, "centerline-vtk-file").c_str());
	file << "# vtk DataFile Version 3.0\nvtk output\nASCII\n";
	file << "DATASET POLYDATA\nPOINTS " << vertices.size() << " int\n";
	for(int i = 0; i < vertices.size(); i++) {
		file << vertices[i].x << " " << vertices[i].y << " " << vertices[i].z << "\n";
	}

	file << "\nLINES " << edges.size() << " " << edges.size()*3 << "\n";
	for(int i = 0; i < edges.size(); i++) {
		file << "2 " << edges[i].x << " " << edges[i].y << "\n";
	}

	file.close();
}
