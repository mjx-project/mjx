cmake_minimum_required(VERSION 3.11)
project(mjx)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "-fPIC")

add_subdirectory(mjx)
add_subdirectory(pybind11)
pybind11_add_module(_mjx src/mjx/pybind.cpp)
target_include_directories(_mjx PRIVATE ${MJX_INCLUDE_DIR})
target_link_libraries(_mjx PRIVATE mjx)
target_compile_definitions(_mjx PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
