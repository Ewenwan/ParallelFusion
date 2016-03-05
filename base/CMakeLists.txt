cmake_minimum_required(VERSION 3.0)

project(ParallelFusion_base)
file(GLOB BASE_HEADER "*.h")
file(GLOB BASE_SOURCE "*.cpp")

add_library(ParallelFusion_base ${BASE_HEADER} ${BASE_SOURCE})

#add third party dependencies when needed