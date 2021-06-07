cmake_minimum_required(VERSION 3.11)
project(mjx_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

###################################################################################
# GoogleTest
###################################################################################
include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        release-1.10.0
)
FetchContent_MakeAvailable(googletest)

###################################################################################
# Define target
###################################################################################
add_executable(mjx_test
        action_test.cpp
        environment_test.cpp
        game_seed_test.cpp
        game_summarizer_test.cpp
        hand_test.cpp
        observation_test.cpp
        open_test.cpp
        environment_test.cpp
        shanten_calculator_test.cpp
        state_test.cpp
        strategy_rule_based_test.cpp
        tile_test.cpp
        type_test.cpp
        utils_test.cpp
        wall_test.cpp
        win_score_test.cpp
        yaku_evaluator_test.cpp
        test_cases.cpp)
target_include_directories(mjx_test PRIVATE ..)
target_link_libraries(mjx_test PRIVATE gtest gtest_main mjx)
target_compile_definitions(mjx_test PRIVATE TEST_RESOURCES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources")
