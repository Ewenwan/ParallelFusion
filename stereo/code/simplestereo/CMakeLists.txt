cmake_minimum_required(VERSION 3.0)

project(SimpleStereo)

find_package(OpenCV REQUIRED)


add_executable(SimpleStereo simplestereo.h simplestereo.cpp SimpleStereo_cli.cpp optimization.h solvers.cpp
        optimization_firstorder.cpp optimization_parallel.cpp optimization_hierarchy.cpp optimize_victor.cpp optimization_multiway.cpp
        stereo_pipeline.h stereo_pipeline.cpp)

target_link_libraries(SimpleStereo ${OpenCV_LIBS}
                      gflags glog
                      ${MRF_LIBS}
                      ${QPBO_LIBS}
                      ${STEREO_BASE_LIBS})
target_include_directories(SimpleStereo PUBLIC
                           ${MRF_DIRS}
                           ${QPBO_DIRS}
                           ${STEREO_BASE_DIRS}
                           ${BASE_DIRS})