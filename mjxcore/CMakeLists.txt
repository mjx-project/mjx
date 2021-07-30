cmake_minimum_required(VERSION 3.11)
project(cmake_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "-fPIC")

add_subdirectory(/Users/sotetsuk/github/mjx/mjx mjx/build)
add_subdirectory(pybind11)
pybind11_add_module(cmake_example src/entry_point.cpp)
target_include_directories(cmake_example PRIVATE ${MJX_INCLUDE_DIR})
target_link_libraries(cmake_example PRIVATE mjx)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(cmake_example PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
