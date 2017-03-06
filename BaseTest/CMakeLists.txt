cmake_minimum_required(VERSION 3.0)

project(BaseTest)


find_package(OpenCV REQUIRED)

file(GLOB SOURCES "*.cpp")
add_executable(BaseTest ${SOURCES})
target_link_libraries(BaseTest ${OpenCV_LIBS} ${CV_UTILS_LIBS})
target_include_directories(BaseTest PUBLIC
                            ${CV_UTILS_DIRS})