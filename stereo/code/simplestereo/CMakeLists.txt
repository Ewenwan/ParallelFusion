cmake_minimum_required(VERSION 3.0)

project(SimpleStereo)

find_package(OpenCV REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIR})

add_executable(SimpleStereo simplestereo.h simplestereo.cpp SimpleStereo_cli.cpp)

target_link_libraries(SimpleStereo ${OpenCV_LIBS} gflags glog MRF)