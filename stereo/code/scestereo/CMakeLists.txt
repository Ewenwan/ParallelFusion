cmake_minimum_required(VERSION 3.0)

project(SceStereo) #second order stereo

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIR})
add_executable(SceStereo scestereo.h scestereo.cpp scestereo_cli.cpp local_matcher.cpp local_matcher.h
        proposal.h proposal.cpp)

target_link_libraries(SceStereo ${OpenCV_LIBS} gflags glog stereo_base segment_ms segment_gb)