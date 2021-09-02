cmake_minimum_required(VERSION 3.11)
project(mjx)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "-fPIC")
set(MJX_EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)
set(MJX_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
message(STATUS "MJX_INCLUDE_DIR=${MJX_INCLUDE_DIR}")

####################################################################################
# Options
####################################################################################

# set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_libs)
option(MJX_USE_SYSTEM_BOOST "Use Boost previously installed." OFF)
option(MJX_USE_SYSTEM_GRPC "Use gRPC previously installed." OFF)

####################################################################################
# For Boost
####################################################################################

# We use boost as header-only
if(MJX_USE_SYSTEM_BOOST)
  message(STATUS "MJX_USE_SYSTEM_BOOST=ON")
  find_package(Boost REQUIRED)
else()
  message(STATUS "MJX_USE_SYSTEM_BOOST=OFF")
  include(${MJX_INCLUDE_DIR}/mjx/cmake/downloaded_boost.cmake)
endif()
message(STATUS "Include Boost_INCLUDE_DIRS=${Boost_INCLUDE_DIRS}")

###################################################################################
# For gRPc
####################################################################################

if(MJX_USE_SYSTEM_GRPC)
  message(STATUS "MJX_USE_SYSTEM_GRPC=ON")
  include(${MJX_INCLUDE_DIR}/mjx/cmake/preinstalled_grpc.cmake)
else()
  message(STATUS "MJX_USE_SYSTEM_GRPC=OFF")
  include(${MJX_INCLUDE_DIR}/mjx/cmake/downloaded_grpc.cmake)
endif()

# Proto file
get_filename_component(mjx_proto "include/mjx/internal/mjx.proto" ABSOLUTE)
get_filename_component(mjx_proto_path "${mjx_proto}" PATH)

# Generated sources
set(mjx_proto_srcs "${MJX_INCLUDE_DIR}/mjx/internal/mjx.pb.cc")
set(mjx_proto_hdrs "${MJX_INCLUDE_DIR}/mjx/internal/mjx.pb.h")
set(mjx_grpc_srcs "${MJX_INCLUDE_DIR}/mjx/internal/mjx.grpc.pb.cc")
set(mjx_grpc_hdrs "${MJX_INCLUDE_DIR}/mjx/internal/mjx.grpc.pb.h")
add_custom_command(
  OUTPUT "${mjx_proto_srcs}" "${mjx_proto_hdrs}" "${mjx_grpc_srcs}" "${mjx_grpc_hdrs}"
  COMMAND ${_PROTOBUF_PROTOC}
  ARGS --grpc_out "${MJX_INCLUDE_DIR}/mjx/internal"
  --cpp_out "${MJX_INCLUDE_DIR}/mjx/internal"
  -I "${mjx_proto_path}"
  --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
  "${mjx_proto}"
  DEPENDS "${mjx_proto}"
)

include_directories(${MJX_INCLUDE_DIR})

####################################################################################
# For spdlog
####################################################################################

# if(NOT EXISTS ${LIB_DIR}/spdlog)
#     find_package(Git REQUIRED)
#     execute_process(
#             COMMAND ${GIT_EXECUTABLE} clone https://github.com/gabime/spdlog.git ${LIB_DIR}/spdlog
#     )
#     execute_process(
#             COMMAND mkdir -p ${LIB_DIR}/spdlog/build
#     )
#     execute_process(
#             COMMAND cmake .. WORKING_DIRECTORY ${LIB_DIR}/spdlog/build
#     )
#     execute_process(
#             COMMAND make -j WORKING_DIRECTORY ${LIB_DIR}/spdlog/build
#     )
# endif()
#
# add_subdirectory(${LIB_DIR}/spdlog)
#
# if(NOT TARGET spdlog)
#     # Stand-alone build
#     find_package(spdlog REQUIRED)
# endif()

####################################################################################
# Build static mjx library
####################################################################################

add_library(
  mjx STATIC
  include/mjx/action.h
  include/mjx/action.cpp
  include/mjx/agent.h
  include/mjx/agent.cpp
  include/mjx/observation.h
  include/mjx/observation.cpp
  include/mjx/state.h
  include/mjx/state.cpp
  include/mjx/hand.h
  include/mjx/hand.cpp
  include/mjx/env.h
  include/mjx/env.cpp
  include/mjx/internal/abstruct_hand.h
  include/mjx/internal/action.h
  include/mjx/internal/agent.h
  include/mjx/internal/agent_batch_grpc_server.h
  include/mjx/internal/agent_batch_local.h
  include/mjx/internal/agent_grpc_client.h
  include/mjx/internal/agent_grpc_server.h
  include/mjx/internal/agent_local.h
  include/mjx/internal/consts.h
  include/mjx/internal/environment.h
  include/mjx/internal/event.h
  include/mjx/internal/game_result_summarizer.h
  include/mjx/internal/game_seed.h
  include/mjx/internal/hand.h
  include/mjx/internal/mjx.h
  include/mjx/internal/observation.h
  include/mjx/internal/open.h
  include/mjx/internal/shanten_cache.h
  include/mjx/internal/shanten_calculator.h
  include/mjx/internal/state.h
  include/mjx/internal/strategy.h
  include/mjx/internal/strategy_rule_based.h
  include/mjx/internal/tile.h
  include/mjx/internal/types.h
  include/mjx/internal/utils.h
  include/mjx/internal/wall.h
  include/mjx/internal/win_cache.h
  include/mjx/internal/win_cache_generator.h
  include/mjx/internal/win_info.h
  include/mjx/internal/win_score.h
  include/mjx/internal/yaku_evaluator.h
  include/mjx/internal/abstruct_hand.cpp
  include/mjx/internal/action.cpp
  include/mjx/internal/agent.cpp
  include/mjx/internal/agent_batch_grpc_server.cpp
  include/mjx/internal/agent_batch_local.cpp
  include/mjx/internal/agent_grpc_client.cpp
  include/mjx/internal/agent_grpc_server.cpp
  include/mjx/internal/agent_local.cpp
  include/mjx/internal/environment.cpp
  include/mjx/internal/event.cpp
  include/mjx/internal/game_result_summarizer.cpp
  include/mjx/internal/game_seed.cpp
  include/mjx/internal/hand.cpp
  include/mjx/internal/observation.cpp
  include/mjx/internal/open.cpp
  include/mjx/internal/shanten_cache.cpp
  include/mjx/internal/shanten_calculator.cpp
  include/mjx/internal/state.cpp
  include/mjx/internal/strategy_rule_based.cpp
  include/mjx/internal/tile.cpp
  include/mjx/internal/types.cpp
  include/mjx/internal/wall.cpp
  include/mjx/internal/win_cache.cpp
  include/mjx/internal/win_cache_generator.cpp
  include/mjx/internal/win_info.cpp
  include/mjx/internal/win_score.cpp
  include/mjx/internal/yaku_evaluator.cpp
  ${mjx_proto_srcs} ${mjx_grpc_srcs} ${mjx_proto_hdrs} ${mjx_grpc_hdrs}
)
target_include_directories(mjx PUBLIC ${Boost_INCLUDE_DIRS} ${MJX_INCLUDE_DIR})
target_link_libraries(mjx PUBLIC ${_REFLECTION} ${_GRPC_GRPCPP} ${_PROTOBUF_LIBPROTOBUF})
target_compile_definitions(mjx PUBLIC WIN_CACHE_DIR="${MJX_INCLUDE_DIR}/mjx/cache")

####################################################################################
# Build cache generators
####################################################################################

# Generate cache file
add_executable(win_cache_generator include/mjx/internal/win_cache_generator.cpp)
target_include_directories(win_cache_generator PUBLIC ${Boost_INCLUDE_DIRS} ${MJX_INCLUDE_DIR})
target_link_libraries(win_cache_generator mjx)
