cmake_minimum_required(VERSION 3.15)
project(OpenMpTasks LANGUAGES CXX)

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 11)

add_executable(OpenMpTasks
        src/com.github.hronosf/2/dot_product.cpp
        src/com.github.hronosf/test.cpp)

target_link_libraries(OpenMpTasks PUBLIC OpenMP::OpenMP_CXX)