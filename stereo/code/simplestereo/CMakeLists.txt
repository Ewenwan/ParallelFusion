cmake_minimum_required(VERSION 3.0)

project(SimpleStereo)

find_package(OpenCV REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIR})

add_executable(SimpleStereo simplestereo.h simplestereo.cpp SimpleStereo_cli.cpp optimization.h solvers.cpp
        optimization_firstorder.cpp optimization_parallel.cpp optimization_hierarchy.cpp optimize_victor.cpp optimization_multiway.cpp
        stereo_pipeline.h stereo_pipeline.cpp)

target_link_libraries(SimpleStereo ${OpenCV_LIBS} gflags glog MRF QPBO1.4 stereo_base)