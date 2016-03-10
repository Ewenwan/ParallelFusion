cmake_minimum_required(VERSION 3.0)

project(LayerDepthMap)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build")
set(CMAKE_BUILD_TYPE Release)

include_directories(/usr/local/include/eigen3)
#add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../base")

find_package(OpenCV REQUIRED)
#find_package(GLOG REQUIRED)
find_package(gflags REQUIRED)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../base ${CMAKE_CURRENT_SOURCE_DIR}/../base/cv_utils)
include_directories(/opt/gurobi650/linux64/include)
set(PROJECT_LINK_LIBS cv_utils.so ParallelFusion_base.a)

#include_directories(../base/cv_utils /opt/gurobi650/linux64/include)
file(GLOB SOURCES "*.cpp")
add_executable(LayerDepthMap ${SOURCES})
target_link_libraries(LayerDepthMap ${OpenCV_LIBS} ${GFlags_LIBS} ${PROJECT_LINK_LIBS} glog gflags)
#target_link_libraries(OcclusionCRF ${OpenGM_LIBS})

#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../base)
