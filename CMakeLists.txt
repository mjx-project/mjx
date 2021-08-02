cmake_minimum_required(VERSION 3.11)
project(mjx-project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "-fPIC")

option(MJX_BUILD_TESTS "Build tests" OFF)

add_subdirectory(mjx)

if(MJX_BUILD_TESTS)
  add_subdirectory(scripts)
  add_subdirectory(tests)
endif()
