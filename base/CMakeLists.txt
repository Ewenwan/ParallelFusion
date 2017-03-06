cmake_minimum_required(VERSION 3.0)

project(ParallelFusion_base)

file(GLOB BASE_HEADER "*.h")
file(GLOB BASE_SOURCE "*.cpp")
include_directories(/usr/local/include/eigen3)

add_executable(ParallelFusion_base ${BASE_HEADER} ${BASE_SOURCE})
target_link_libraries(ParallelFusion_base gflags glog)
set(BASE_LIBS ParallelFusion_base PARENT_SCOPE)
set(BASE_DIRS /usr/local/include/eigen3
    ${CMAKE_CURRENT_SOURCE_DIR}
    PARENT_SCOPE)

#add third party dependencies when needed