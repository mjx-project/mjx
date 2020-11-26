cmake_minimum_required(VERSION 3.9)
project(mahjong)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall")
# set(CMAKE_CXX_FLAGS "--pedantic-errors")

set(SOURCE_FILES main.cpp)
add_executable(${PROJECT_NAME}.out ${SOURCE_FILES})

include_directories(.)

add_subdirectory(external_libs/spdlog)
add_subdirectory(mj)
add_subdirectory(test)
add_subdirectory(dataset)

target_link_libraries(${PROJECT_NAME}.out mj)

