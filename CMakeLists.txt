cmake_minimum_required(VERSION 3.11)
project(mjx-project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "-fPIC")
set(MJX_EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)
set(MJX_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(mjx)
add_subdirectory(scripts)
add_subdirectory(tests)
add_subdirectory(pymjx/_mjx)
