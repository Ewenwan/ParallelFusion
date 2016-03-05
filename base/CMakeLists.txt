cmake_minimum_required(VERSION 3.0)

project(ParallelFusion_base)
file(GLOB BASE_HEADER "*.h")
file(GLOB BASE_SOURCE "*.cpp")

set(CMAKE_CXX_FLAGS "${CMAKE_FXX_FLAGS} -std=c++11 -O3")

add_library(ParallelFusion_base SHARED ${BASE_HEADER} ${BASE_SOURCE})

#add third party dependencies when needed