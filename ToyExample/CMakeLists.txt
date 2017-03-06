cmake_minimum_required(VERSION 3.0)

project(ToyExample)


#set(BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build")

include_directories(/usr/local/include/eigen3)
#add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../base")

find_package(OpenCV REQUIRED)
#find_package(GLOG REQUIRED)
#find_package(GFlags REQUIRED)

#link_directories(../base)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../base ${CMAKE_CURRENT_SOURCE_DIR}/../base/cv_utils)
set(PROJECT_LINK_LIBS cv_utils.so ParallelFusion_base.so)

#set(CMAKE_BUILD_TYPE Release)
#include_directories(../base/cv_utils /opt/gurobi650/linux64/include)
file(GLOB SOURCES "*.cpp")
add_executable(ToyExample ${SOURCES})
target_link_libraries(ToyExample ${OpenCV_LIBS} ${PROJECT_LINK_LIBS})
#target_link_libraries(OcclusionCRF ${OpenGM_LIBS})

#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../base)
