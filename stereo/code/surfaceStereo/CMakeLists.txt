cmake_minimum_required(VERSION 3.0)

project(SurfaceStereo)

file(GLOB surface_src code/*.h code/*.cpp)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${BUILD_DIR}/surfaceStereo")

add_executable(SurfaceStereo ${surface_src})

target_link_libraries(SurfaceStereo ${OpenCV_LIBS})