project(mjx_scripts)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(speed_benchmark speed_benchmark.cpp)
target_include_directories(speed_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(speed_benchmark PRIVATE mjx)
