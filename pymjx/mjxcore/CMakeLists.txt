cmake_minimum_required(VERSION 3.11)
project(_mjx)

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

add_subdirectory($ENV{MJX_DIR} mjx/build)
pybind11_add_module(_mjx src/entry_point.cpp)
target_include_directories(_mjx PRIVATE ${MJX_INCLUDE_DIR})
target_link_libraries(_mjx PRIVATE mjx)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(_mjx PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
