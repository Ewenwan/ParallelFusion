cmake_minimum_required(VERSION 3.0)
project (cv_utils)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS features kdtree)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

file(GLOB SOURCES "*.cpp")
set(CV_UTILS_DIRS ${TRW_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(cv_utils SHARED ${SOURCES})
target_link_libraries(cv_utils ${OpenCV_LIBS} ${TRW_LIBS})
target_link_libraries(cv_utils ${PCL_FEATURES_LIBRARIES} ${PCL_KDTREE_LIBRARIES})
target_include_directories(cv_utils PUBLIC
                           ${CV_UTILS_DIRS}
                           ${BASE_DIRS})
set(CV_UTILS_LIBS cv_utils PARENT_SCOPE)
set(CV_UTILS_DIRS ${CV_UTILS_DIRS} PARENT_SCOPE)