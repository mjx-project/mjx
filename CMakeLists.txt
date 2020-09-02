cmake_minimum_required(VERSION 3.9)
project(mahjong)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall")
# set(CMAKE_CXX_FLAGS "--pedantic-errors")

set(SOURCE_FILES main.cpp)
add_executable(${CMAKE_PROJECT_NAME}.out ${SOURCE_FILES})

include_directories(mahjong)
include_directories(${CMAKE_BINARY_DIR}/mahjong)  # to include generated gRPC files

add_subdirectory(mahjong)
add_subdirectory(test)

target_link_libraries(${CMAKE_PROJECT_NAME}.out ${CMAKE_PROJECT_NAME})

