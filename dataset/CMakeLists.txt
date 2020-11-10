project(mj_dataset)

add_executable(generate_dataset train_data_generator.cpp)
target_link_libraries(generate_dataset mj)

add_definitions(-DRESOURCES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources")
