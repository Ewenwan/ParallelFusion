cmake_minimum_required(VERSION 3.0)

project(ParallelFusion)

set(CMAKE_CXX_FLAGS "${CMAKE_FXX_FLAGS} -std=c++11 -O3")

add_subdirectory(base)
add_subdirectory(stereo)

#uncomment this line one we have something in opticalflow
#add_subdirectory(opticalflow)