cmake_minimum_required(VERSION 3.0)

project(stereo_base)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BUILD_DIR})
add_library(stereo_base file_io.h configurator.cpp configurator.h plane3D.cpp plane3D.h utility.cpp utility.h
        depth.cpp depth.h local_matcher.h local_matcher.cpp)

target_link_libraries(stereo_base gflags glog)