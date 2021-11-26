cmake_minimum_required(VERSION 3.11)
project(mjx_tests_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

###################################################################################
# GoogleTest
###################################################################################
find_package(GTest)
if(NOT GTest_FOUND)
  include(FetchContent)
  set(FETCHCONTENT_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)
  set(FETCHCONTENT_QUIET OFF)
  set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
  fetchcontent_declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.11.0
    GIT_PROGRESS TRUE
  )
  fetchcontent_makeavailable(googletest)
endif()

###################################################################################
# Define target
###################################################################################
add_executable(
  mjx_tests_cpp
  action_test.cpp
  event_test.cpp
  env_test.cpp
  hand_test.cpp
  observation_test.cpp
  state_test.cpp
  seed_generator_test.cpp
  internal_action_test.cpp
  internal_environment_test.cpp
  internal_game_seed_test.cpp
  internal_game_result_summarizer_test.cpp
  internal_hand_test.cpp
  internal_observation_test.cpp
  internal_open_test.cpp
  internal_environment_test.cpp
  internal_shanten_calculator_test.cpp
  internal_state_test.cpp
  internal_strategy_rule_based_test.cpp
  internal_tile_test.cpp
  internal_type_test.cpp
  internal_utils_test.cpp
  internal_wall_test.cpp
  internal_win_score_test.cpp
  internal_yaku_evaluator_test.cpp
  failure_cases_test.cpp
)
target_include_directories(mjx_tests_cpp PRIVATE ${MJX_INCLUDE_DIR})
target_link_libraries(mjx_tests_cpp PRIVATE gtest gtest_main mjx)
target_compile_definitions(mjx_tests_cpp PRIVATE TEST_RESOURCES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources")
