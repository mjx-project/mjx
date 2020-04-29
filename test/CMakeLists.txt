project(${CMAKE_PROJECT_NAME}_test)

# Clone Google Test
set(GOOGLETEST_DIR ${CMAKE_SOURCE_DIR}/test/lib/googletest)
if(NOT EXISTS ${GOOGLETEST_DIR})
    find_package(Git REQUIRED)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} clone https://github.com/google/googletest ${GOOGLETEST_DIR}
    )
endif()

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
add_subdirectory(lib/googletest)

file(GLOB TEST_FILES *.cpp)
add_executable(${CMAKE_PROJECT_NAME}_test ${TEST_FILES})

target_link_libraries(${CMAKE_PROJECT_NAME}_test gtest gtest_main)
target_link_libraries(${CMAKE_PROJECT_NAME}_test ${CMAKE_PROJECT_NAME})
