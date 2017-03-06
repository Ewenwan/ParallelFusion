cmake_minimum_required(VERSION 3.0)

project(SceStereo) #second order stereo

add_executable(SceStereo scestereo.h scestereo.cpp scestereo_cli.cpp
        proposal.h proposal.cpp optimization.h optimization_fusionmove.cpp optimization_firstorder.cpp)

target_link_libraries(SceStereo ${OpenCV_LIBS}
                      gflags
                      glog
                      stereo_base
                      segment_ms
                      segment_gb
                      ${QPBO_LIBS}
                      ${MRF_LIBS}
                      ${STEREO_BASE_LIBS})
target_include_directories(SceStereo PUBLIC
                           ${QPBO_DIRS}
                           ${MRF_DIRS}
                           ${BASE_DIRS}
                           ${STEREO_BASE_DIRS})