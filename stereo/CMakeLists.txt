cmake_minimum_required(VERSION 3.0)

project(stereo)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
set(BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build")

include_directories(/usr/local/include/eigen3)

find_package(OpenCV REQUIRED)

#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/code/surfaceStereo)
