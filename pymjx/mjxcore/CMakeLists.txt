cmake_minimum_required(VERSION 3.11)
project(mjxcore)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "-fPIC")

# download pybind11
include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${MJX_EXTERNAL_DIR})
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
fetchcontent_declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.6.2
)
fetchcontent_makeavailable(pybind11)

add_subdirectory(/Users/sotetsuk/github/mjx/mjx mjx/build)
pybind11_add_module(mjxcore src/entry_point.cpp)
target_include_directories(mjxcore PRIVATE ${MJX_INCLUDE_DIR})
target_link_libraries(mjxcore PRIVATE mjx)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(mjxcore PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
