cmake_minimum_required(VERSION 3.0)

project(LayerDepthMap)


set(CMAKE_BUILD_TYPE Release)


find_package(OpenCV REQUIRED)
#find_package(GLOG REQUIRED)
find_package(gflags REQUIRED)


file(GLOB SOURCES "*.cpp")

add_executable(LayerDepthMap ${SOURCES})
target_link_libraries(LayerDepthMap ${OpenCV_LIBS}
                      ${GFlags_LIBS}
                      glog gflags
                      ${TRW_LIBS}
                      ${CV_UTILS_LIBS})
target_include_directories(LayerDepthMap PUBLIC
                           ${TRW_DIRS}
                           ${CV_UTILS_LIBS})

