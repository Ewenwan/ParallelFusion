cmake_minimum_required(VERSION 3.0)

project(BaseTest)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")
#set(BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build")

#include_directories(/usr/local/include/eigen3)
#add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../base")

find_package(OpenCV REQUIRED)
#find_package(GLOG REQUIRED)
#find_package(GFlags REQUIRED)

#link_directories(../base)

#set(CMAKE_BUILD_TYPE Release)
#include_directories(../base/cv_utils /opt/gurobi650/linux64/include)
file(GLOB SOURCES "*.cpp")
add_executable(BaseTest ${SOURCES})
target_link_libraries(BaseTest ${OpenCV_LIBS} ${CV_UTILS})
#target_link_libraries(OcclusionCRF ${OpenGM_LIBS})
#target_link_libraries(ToyExample ${PROJECT_LINK_LIBS})

#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../base)
