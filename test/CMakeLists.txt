# Clone Google Test
set(GOOGLETEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/googletest)
if(NOT EXISTS ${GOOGLETEST_DIR})
    find_package(Git REQUIRED)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} clone https://github.com/google/googletest ${GOOGLETEST_DIR}
    )
endif()

include_directories(${CMAKE_BINARY_DIR}/mahjong)  # to include generated gRPC files
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
add_subdirectory(${GOOGLETEST_DIR})

file(GLOB TEST_FILES *.cpp)
add_executable(${PROJECT_NAME}_test ${TEST_FILES})

target_link_libraries(${PROJECT_NAME}_test gtest gtest_main)
target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME})

# Path to test resources
add_definitions(-DTEST_RESOURCES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources")