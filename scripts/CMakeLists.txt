project(mjx_scripts)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(speed_benchmark speed_benchmark.cpp)
add_executable(run_server run_server.cpp)
add_executable(run_games run_games.cpp)
target_include_directories(speed_benchmark PRIVATE ${MJX_INCLUDE_DIR})
target_link_libraries(speed_benchmark PRIVATE mjx)
target_link_libraries(run_server PRIVATE mjx)
target_link_libraries(run_games PRIVATE mjx)
