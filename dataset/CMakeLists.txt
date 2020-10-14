project(mj_dataset)

add_executable(${PROJECT_NAME} train_data_generator.cpp)
target_link_libraries(${PROJECT_NAME} mj)

add_definitions(-DRESOURCES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources")
