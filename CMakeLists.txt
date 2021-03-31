cmake_minimum_required(VERSION 3.11)
project(mjx-project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall")
# set(CMAKE_CXX_FLAGS "--pedantic-errors")

set(CMAKEDIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(EXTERNALDIR ${CMAKE_CURRENT_SOURCE_DIR}/external)

include(${CMAKEDIR}/distributions.cmake)

set(SOURCE_FILES main.cpp)
add_executable(main.out ${SOURCE_FILES})

include_directories(.)

add_subdirectory(mjx)
add_subdirectory(scripts)
add_subdirectory(tests)

target_link_libraries(main.out mjx)

