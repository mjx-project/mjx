project(mjx_scripts)

    add_executable(speed_benchmark speed_benchmark.cpp)

        target_link_libraries(speed_benchmark mjx)
