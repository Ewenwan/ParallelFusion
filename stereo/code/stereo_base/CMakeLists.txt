cmake_minimum_required(VERSION 3.0)

project(stereo_base)

add_library(stereo_base file_io.h configurator.cpp configurator.h plane3D.cpp plane3D.h utility.cpp utility.h
        depth.cpp depth.h local_matcher.h local_matcher.cpp)

target_link_libraries(stereo_base gflags glog)
set(STEREO_BASE_LIBS stereo_base PARENT_SCOPE)
set(STEREO_BASE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIRS}
    PARENT_SCOPE)