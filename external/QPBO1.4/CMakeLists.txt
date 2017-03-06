cmake_minimum_required(VERSION 3.0)

project(QPBO1.4)

file(GLOB HDR block.h QPBO.h ELC.h ELC0.h)
file(GLOB SRC QPBO.cpp QPBO_extra.cpp QPBO_maxflow.cpp QPBO_postprocessing.cpp)

add_library(QPBO1.4 ${HDR} ${SRC})
set(QPBO_LIBS QPBO1.4 PARENT_SCOPE)
set(QPBO_DIRS ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)