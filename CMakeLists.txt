cmake_minimum_required(VERSION 3.11)
project(mjx-project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "-fPIC")

option(MJX_BUILD_TESTS "Build tests" OFF)
option(MJX_BUILD_PYTHON "Build as Python liblary " OFF)

add_subdirectory(mjx)

if(MJX_BUILD_TESTS)
  add_subdirectory(tests)
  add_subdirectory(scripts)
endif()

if(MJX_BUILD_PYTHON)
  add_subdirectory(pybind11)
  pybind11_add_module(_mjx pymjx/mjx/pybind.cpp)
  target_include_directories(_mjx PRIVATE ${MJX_INCLUDE_DIR})
  target_link_libraries(_mjx PRIVATE mjx)
  target_compile_definitions(_mjx PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
endif()
