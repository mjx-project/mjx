project(mjx_test)
include(${CMAKEDIR}/googletest.cmake)

file(GLOB TEST_FILES *.cpp)
add_executable(${PROJECT_NAME} ${TEST_FILES})

target_link_libraries(${PROJECT_NAME} gtest gtest_main)
target_link_libraries(${PROJECT_NAME} mjx)

# Path to test resources
add_definitions(-DTEST_RESOURCES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources")
