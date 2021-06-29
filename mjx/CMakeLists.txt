cmake_minimum_required(VERSION 3.11)
project(mjx)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

####################################################################################
# Options
####################################################################################

# set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_libs)
option(MJX_USE_SYSTEM_BOOST "Use Boost previously installed." ON)
option(MJX_USE_SYSTEM_GRPC "Use gRPC previously installed." ON)

####################################################################################
# For Boost
####################################################################################

# We use boost as header-only
if (MJX_USE_SYSTEM_BOOST)
    message(STATUS "MJX_USE_SYSTEM_BOOST=ON")
    find_package(Boost REQUIRED)
else()
    message(STATUS "MJX_USE_SYSTEM_BOOST=OFF")
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/downloaded_boost.cmake)
endif()
message(STATUS "Include Boost_INCLUDE_DIRS=${Boost_INCLUDE_DIRS}")

###################################################################################
# For gRPc
####################################################################################

if (MJX_USE_SYSTEM_GRPC)
    message(STATUS "MJX_USE_SYSTEM_GRPC=ON")
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/preinstalled_grpc.cmake)
else()
    message(STATUS "MJX_USE_SYSTEM_GRPC=OFF")
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/downloaded_grpc.cmake)
endif()

# Proto file
get_filename_component(mjx_proto "internal/mjx.proto" ABSOLUTE)
get_filename_component(mjx_proto_path "${mjx_proto}" PATH)

# Generated sources
set(mjx_proto_srcs "${CMAKE_CURRENT_SOURCE_DIR}/internal/mjx.pb.cc")
set(mjx_proto_hdrs "${CMAKE_CURRENT_SOURCE_DIR}/internal/mjx.pb.h")
set(mjx_grpc_srcs "${CMAKE_CURRENT_SOURCE_DIR}/internal/mjx.grpc.pb.cc")
set(mjx_grpc_hdrs "${CMAKE_CURRENT_SOURCE_DIR}/internal/mjx.grpc.pb.h")
add_custom_command(
        OUTPUT "${mjx_proto_srcs}" "${mjx_proto_hdrs}" "${mjx_grpc_srcs}" "${mjx_grpc_hdrs}"
        COMMAND ${_PROTOBUF_PROTOC}
        ARGS --grpc_out "${CMAKE_CURRENT_SOURCE_DIR}/internal"
        --cpp_out "${CMAKE_CURRENT_SOURCE_DIR}/internal"
        -I "${mjx_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${mjx_proto}"
        DEPENDS "${mjx_proto}")

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

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

add_library(mjx STATIC
        env.h
        env.cpp
        pyenv.h
        pyenv.cpp
        internal/abstruct_hand.h
        internal/action.h
        internal/agent.h
        internal/agent_batch_grpc_server.h
        internal/agent_batch_local.h
        internal/agent_grpc_client.h
        internal/agent_grpc_server.h
        internal/agent_local.h
        internal/consts.h
        internal/environment.h
        internal/event.h
        internal/game_result_summarizer.h
        internal/game_seed.h
        internal/hand.h
        internal/mjx.h
        internal/observation.h
        internal/open.h
        internal/shanten_cache.h
        internal/shanten_calculator.h
        internal/state.h
        internal/strategy.h
        internal/strategy_rule_based.h
        internal/tile.h
        internal/types.h
        internal/utils.h
        internal/wall.h
        internal/win_cache.h
        internal/win_cache_generator.h
        internal/win_info.h
        internal/win_score.h
        internal/yaku_evaluator.h
        internal/abstruct_hand.cpp
        internal/action.cpp
        internal/agent.cpp
        internal/agent_batch_grpc_server.cpp
        internal/agent_batch_local.cpp
        internal/agent_grpc_client.cpp
        internal/agent_grpc_server.cpp
        internal/agent_local.cpp
        internal/environment.cpp
        internal/event.cpp
        internal/game_result_summarizer.cpp
        internal/game_seed.cpp
        internal/hand.cpp
        internal/observation.cpp
        internal/open.cpp
        internal/shanten_cache.cpp
        internal/shanten_calculator.cpp
        internal/state.cpp
        internal/strategy_rule_based.cpp
        internal/tile.cpp
        internal/types.cpp
        internal/wall.cpp
        internal/win_cache.cpp
        internal/win_cache_generator.cpp
        internal/win_info.cpp
        internal/win_score.cpp
        internal/yaku_evaluator.cpp
        ${mjx_proto_srcs} ${mjx_grpc_srcs} ${mjx_proto_hdrs} ${mjx_grpc_hdrs})
target_include_directories(mjx PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(mjx PUBLIC ${_REFLECTION} ${_GRPC_GRPCPP} ${_PROTOBUF_LIBPROTOBUF})
target_compile_definitions(mjx PUBLIC WIN_CACHE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/cache")

####################################################################################
# Build cache generators
####################################################################################

# Generate cache file
add_executable(win_cache_generator internal/win_cache_generator.cpp)
target_link_libraries(win_cache_generator ${PROJECT_NAME})
