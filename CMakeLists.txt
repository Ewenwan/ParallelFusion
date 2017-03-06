cmake_minimum_required(VERSION 3.0)

project(ParallelFusion)

set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/base)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})


set(CMAKE_CXX_FLAGS "-std=gnu++14 -O2 -g")
add_subdirectory(external)
add_subdirectory(base)
add_subdirectory(base/cv_utils)


add_subdirectory(LayerDepthMap)
add_subdirectory(BaseTest)
add_subdirectory(stereo)

add_subdirectory(AlphaMatting)


#uncomment this line one we have something in opticalflow
add_subdirectory(OpticalFlow)