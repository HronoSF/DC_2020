cmake_minimum_required(VERSION 3.15)
project(OpenMpTasks LANGUAGES CXX)

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 11)

add_executable(OpenMpTasks
        src/com.github.hronosf/run.cpp
        src/com.github.hronosf/1/MinVectorElementTask.cpp
        src/com.github.hronosf/2/VectorDotProductTask.cpp
        src/com.github.hronosf/3/IntergalValueTask.cpp
        src/com.github.hronosf/4/MaxMinMatrixValue.cpp
        src/com.github.hronosf/6/ReductionTask.cpp
        src/com.github.hronosf/OpenMPTask.hpp)


target_link_libraries(OpenMpTasks PUBLIC OpenMP::OpenMP_CXX)