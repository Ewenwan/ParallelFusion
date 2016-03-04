cmake_minimum_required(VERSION 3.0)

project(SurfaceStereo)

find_package(OpenCV REQUIRED)

file(GLOB surface_src code/*.h code/*.cpp)

add_executable(SurfaceStereo ${surface_src})

target_link_libraries(SurfaceStereo ${OpenCV_LIBS})