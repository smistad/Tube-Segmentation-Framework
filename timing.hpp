#ifdef CPP11
#include <chrono>
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
#define INIT_TIMER high_resolution_clock::time_point timerStart = high_resolution_clock::now();
#define START_TIMER  timerStart = high_resolution_clock::now();
#define STOP_TIMER(name)  std::cout << "RUNTIME of " << name << ": " << \
        duration_cast<milliseconds>( \
                            high_resolution_clock::now()-timerStart \
                    ).count() << " ms " << std::endl;
#else
#define INIT_TIMER
#define START_TIMER
#define STOP_TIMER(name)
#endif
