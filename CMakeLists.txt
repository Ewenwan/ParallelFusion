cmake_minimum_required(VERSION 3.0)

project(ParallelFusion)

set(CMAKE_CXX_FLAGS "${CMAKE_FXX_FLAGS} -std=c++14 -O3")
set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/base)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})



add_subdirectory(base)
add_subdirectory(base/cv_utils)

add_subdirectory(LayerDepthMap)
add_subdirectory(BaseTest)
add_subdirectory(stereo)
add_subdirectory(ToyExample)


#uncomment this line one we have something in opticalflow
add_subdirectory(OpticalFlow)