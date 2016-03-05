cmake_minimum_required(VERSION 3.0)

project(stereo_base)

add_library(stereo_base file_io.h configurator.cpp configurator.h plane3D.cpp plane3D.h utility.cpp utility.h)

target_link_libraries(stereo_base gflags glog)