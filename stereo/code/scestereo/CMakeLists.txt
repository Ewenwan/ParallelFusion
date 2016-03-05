cmake_minimum_required(VERSION 3.0)

project(SceStereo) #second order stereo

add_executable(SceStereo scestereo.h scestereo.cpp scestereo_cli.cpp local_matcher.cpp local_matcher.h)

target_link_libraries(SceStereo ${OpenCV_LIBS} gflags glog stereo_base)