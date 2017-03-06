cmake_minimum_required(VERSION 3.0)
add_subdirectory(segment_ms)
add_subdirectory(segment_gb)
add_subdirectory(QPBO1.4)
set(QPBO_LIBS ${QPBO_LIBS} PARENT_SCOPE)
set(QPBO_DIRS ${QPBO_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    PARENT_SCOPE)
add_subdirectory(MRF2.2)
set(MRF_LIBS ${MRF_LIBS} PARENT_SCOPE)
set(MRF_DIRS ${MRF_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    PARENT_SCOPE)


add_subdirectory(TRW_S)
set(TRW_LIBS ${TRW_LIBS} PARENT_SCOPE)
set(TRW_DIRS ${TRW_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    PARENT_SCOPE)