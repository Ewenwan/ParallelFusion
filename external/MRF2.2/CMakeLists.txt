cmake_minimum_required(VERSION 3.0)

project(MRF)

file(GLOB SRC "*.cpp")
file(GLOB HDR "*.h")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -DUSE_64_BIT_PTR_CAST")

add_library(MRF ${SRC} ${HDR})
target_link_libraries(MRF glog)

set(MRF_LIBS MRF PARENT_SCOPE)
set(MRF_DIRS ${CMAKE_CURRENT_SOURCE_DIRS} PARENT_SCOPE)