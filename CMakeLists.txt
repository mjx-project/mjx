cmake_minimum_required(VERSION 3.11)
project(mjx-project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "-fPIC")

add_subdirectory(mjx)
add_subdirectory(scripts)
add_subdirectory(tests)


# pybind 11
include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
fetchcontent_declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.6.2
)
fetchcontent_makeavailable(pybind11)

pybind11_add_module(pyenv entry_point.cpp)
target_link_libraries(pyenv PRIVATE mjx)
