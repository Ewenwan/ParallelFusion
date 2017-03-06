cmake_minimum_required(VERSION 3.0)

project(stereo)

set(CMAKE_BUILD_TYPE Release)

include_directories(/usr/local/include/eigen3)

find_package(OpenCV REQUIRED)
find_package(gflags REQUIRED)

unset(OPENMP_FOUND)
find_package(OpenMP QUIET)

if(OPENMP_FOUND)
    message("-- Found OpenMP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message("-- Compiler flags: ${CMAKE_CXX_FLAGS}")
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/code/stereo_base)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/code/scestereo)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/code/simplestereo)
