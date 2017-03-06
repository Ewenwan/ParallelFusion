cmake_minimum_required(VERSION 3.0)

project(OpticalFlow)


find_package(OpenCV REQUIRED)
#find_package(GLOG REQUIRED)
find_package(gflags REQUIRED)


file(GLOB SOURCES "*.cpp")
add_executable(OpticalFlow ${SOURCES})
target_link_libraries(OpticalFlow ${OpenCV_LIBS} ${GFlags_LIBS}
                      glog
                      gflags -lpthread
                      ${TRW_LIBS})
target_include_directories(OpticalFlow PUBLIC
                           ${TRW_DIRS}
                           ${BASE_DIRS})

