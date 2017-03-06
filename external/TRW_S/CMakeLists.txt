cmake_minimum_required(VERSION 3.0)

project(TRW_S)

file(GLOB SRC "*.cpp")
file(GLOB HDR "*.h")

add_library(TRW_S ${SRC} ${HDR})
set(TRW_LIBS TRW_S PARENT_SCOPE)
set(TRW_DIRS ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)