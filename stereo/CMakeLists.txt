cmake_minimum_required(VERSION 3.0)

project(stereo)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
set(BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build")

include_directories(/usr/local/include/eigen3)

find_package(OpenCV REQUIRED)
find_package(gflags REQUIRED)

#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/code/surfaceStereo)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/code/scestereo)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/code/stereo_base)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/code/external/segment_ms)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/code/external/segment_gb)
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external)