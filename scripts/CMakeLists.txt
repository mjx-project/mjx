project(mjx_scripts)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(speed_benchmark speed_benchmark.cpp)
target_include_directories(speed_benchmark PRIVATE ${MJX_INCLUDE_DIR})
target_link_libraries(speed_benchmark PRIVATE mjx)

add_executable(generate_feature generate_feature.cpp)
target_include_directories(generate_feature PRIVATE ${MJX_INCLUDE_DIR})
target_link_libraries(generate_feature PRIVATE mjx)
