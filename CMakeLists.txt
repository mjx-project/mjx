cmake_minimum_required(VERSION 3.9)
project(mjx_repo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall")
# set(CMAKE_CXX_FLAGS "--pedantic-errors")

set(SOURCE_FILES main.cpp)
add_executable(main.out ${SOURCE_FILES})

include_directories(.)

add_subdirectory(mjx)
add_subdirectory(scripts)
add_subdirectory(test)

target_link_libraries(main.out mjx)

