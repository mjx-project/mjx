cmake_minimum_required(VERSION 3.11)
project(mjx_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

###################################################################################
# GoogleTest
###################################################################################
include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        release-1.10.0
)
FetchContent_MakeAvailable(googletest)

###################################################################################
# Define target
###################################################################################
file(GLOB TEST_FILES *.cpp)
add_executable(mjx_test ${TEST_FILES})
target_include_directories(mjx_test PRIVATE ..)
target_link_libraries(mjx_test PRIVATE mjx)
target_link_libraries(mjx_test PRIVATE gtest gtest_main)
target_compile_definitions(mjx_test PRIVATE TEST_RESOURCES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources")
