cmake_minimum_required(VERSION 3.11)
project(mjx-project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "-fPIC")
# set(CMAKE_CXX_FLAGS "--pedantic-errors")

add_subdirectory(mjx)
add_subdirectory(scripts)
add_subdirectory(tests)

add_subdirectory(pybind11)
pybind11_add_module(pyenv entry_point.cpp)
target_link_libraries(pyenv PRIVATE mjx)