cmake_minimum_required(VERSION 3.0)

project(ParallelFusion_base)

set(CMAKE_CXX_FLAGS "${CMAKE_FXX_FLAGS} -std=c++11 -O3")

file(GLOB BASE_HEADER "*.h")
file(GLOB BASE_SOURCE "*.cpp")
include_directories(/usr/local/include/eigen3)

add_executable(ParallelFusion_base ${BASE_HEADER} ${BASE_SOURCE})
target_link_libraries(ParallelFusion_base gflags glog)

#add third party dependencies when needed