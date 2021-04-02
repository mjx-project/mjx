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

set(MJX_EXTERNALDIR ${CMAKE_CURRENT_SOURCE_DIR}/external)

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
get_filename_component(mjx_proto "../mjx.proto" ABSOLUTE)
get_filename_component(mjx_proto_path "${mjx_proto}" PATH)

# Generated sources
set(mjx_proto_srcs "${CMAKE_CURRENT_SOURCE_DIR}/mjx.pb.cc")
set(mjx_proto_hdrs "${CMAKE_CURRENT_SOURCE_DIR}/mjx.pb.h")
set(mjx_grpc_srcs "${CMAKE_CURRENT_SOURCE_DIR}/mjx.grpc.pb.cc")
set(mjx_grpc_hdrs "${CMAKE_CURRENT_SOURCE_DIR}/mjx.grpc.pb.h")
add_custom_command(
        OUTPUT "${mjx_proto_srcs}" "${mjx_proto_hdrs}" "${mjx_grpc_srcs}" "${mjx_grpc_hdrs}"
        COMMAND ${_PROTOBUF_PROTOC}
        ARGS --grpc_out "${CMAKE_CURRENT_SOURCE_DIR}"
        --cpp_out "${CMAKE_CURRENT_SOURCE_DIR}"
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
        abstruct_hand.h
        action.h
        agent.h
        agent_batch_grpc_server.h
        agent_batch_local.h
        agent_grpc_client.h
        agent_grpc_server.h
        agent_local.h
        consts.h
        environment.h
        event.h
        game_result_summarizer.h
        game_seed.h
        hand.h
        mjx.h
        observation.h
        open.h
        shanten_cache.h
        shanten_cache_generator.h
        shanten_calculator.h
        state.h
        strategy.h
        strategy_rule_based.h
        tile.h
        types.h
        utils.h
        wall.h
        win_cache.h
        win_cache_generator.h
        win_info.h
        win_score.h
        yaku_evaluator.h
        abstruct_hand.cpp
        action.cpp
        agent.cpp
        agent_batch_grpc_server.cpp
        agent_batch_local.cpp
        agent_grpc_client.cpp
        agent_grpc_server.cpp
        agent_local.cpp
        environment.cpp
        event.cpp
        game_result_summarizer.cpp
        game_seed.cpp
        hand.cpp
        observation.cpp
        open.cpp
        shanten_cache.cpp
        shanten_cache_generator.cpp
        shanten_calculator.cpp
        state.cpp
        strategy_rule_based.cpp
        tile.cpp
        types.cpp
        wall.cpp
        win_cache.cpp
        win_cache_generator.cpp
        win_info.cpp
        win_score.cpp
        yaku_evaluator.cpp
        ${mjx_proto_srcs} ${mjx_grpc_srcs} ${mjx_proto_hdrs} ${mjx_grpc_hdrs})
target_include_directories(mjx PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(mjx PUBLIC ${_REFLECTION} ${_GRPC_GRPCPP} ${_PROTOBUF_LIBPROTOBUF})
target_compile_definitions(mjx PUBLIC WIN_CACHE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/cache")

####################################################################################
# Build cache generators
####################################################################################

# Generate cache file
add_executable(win_cache_generator win_cache_generator.cpp)
target_link_libraries(win_cache_generator ${PROJECT_NAME})

add_executable(shanten_cache_generator shanten_cache_generator.cpp)
target_link_libraries(shanten_cache_generator ${PROJECT_NAME})
