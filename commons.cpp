#include "commons.hpp"

void TSFGarbageCollector::addMemObject(cl::Memory* mem) {
    memObjects.insert(mem);
}

void TSFGarbageCollector::deleteMemObject(cl::Memory* mem) {
    memObjects.erase(mem);
    delete mem;
    mem = NULL;
}

void TSFGarbageCollector::deleteAllMemObjects() {
    std::set<cl::Memory *>::iterator it;
    for(it = memObjects.begin(); it != memObjects.end(); it++) {
        cl::Memory * mem = *it;
        delete (mem);
        mem = NULL;
    }
    memObjects.clear();
}

TSFGarbageCollector::~TSFGarbageCollector() {
}
