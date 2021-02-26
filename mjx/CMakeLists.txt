project(mjx)
set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_libs)

####################################################################################
# For Boost
####################################################################################
find_package(Boost)
IF (Boost_FOUND)
    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
    ADD_DEFINITIONS( "-DHAS_BOOST" )
ENDIF()

####################################################################################
# For gRPc
####################################################################################
# Assumes that gRPC dependencies are previously installed (and thus following codes can find them)
find_package(Threads REQUIRED)
set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${protobuf_VERSION}")
set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
set(_REFLECTION gRPC::grpc++_reflection)
find_program(_PROTOBUF_PROTOC protoc)
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")
set(_GRPC_GRPCPP gRPC::grpc++)
find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)

# Proto file
get_filename_component(mj_proto "../protos/mj.proto" ABSOLUTE)
get_filename_component(mj_proto_path "${mj_proto}" PATH)

# Generated sources
set(mj_proto_srcs "${CMAKE_CURRENT_SOURCE_DIR}/mj.pb.cc")
set(mj_proto_hdrs "${CMAKE_CURRENT_SOURCE_DIR}/mj.pb.h")
set(mj_grpc_srcs "${CMAKE_CURRENT_SOURCE_DIR}/mj.grpc.pb.cc")
set(mj_grpc_hdrs "${CMAKE_CURRENT_SOURCE_DIR}/mj.grpc.pb.h")
add_custom_command(
        OUTPUT "${mj_proto_srcs}" "${mj_proto_hdrs}" "${mj_grpc_srcs}" "${mj_grpc_hdrs}"
        COMMAND ${_PROTOBUF_PROTOC}
        ARGS --grpc_out "${CMAKE_CURRENT_SOURCE_DIR}"
        --cpp_out "${CMAKE_CURRENT_SOURCE_DIR}"
        -I "${mj_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${mj_proto}"
        DEPENDS "${mj_proto}")

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

# Targets for quick gRPC execution test

# foreach(_target agent_server_mock agent_client)
#     add_executable(${_target} "${_target}.cpp"
#             ${mj_proto_srcs}
#             ${mj_grpc_srcs})
#     target_link_libraries(${_target}
#             ${_REFLECTION}
#             ${_GRPC_GRPCPP}
#             ${_PROTOBUF_LIBPROTOBUF})
# endforeach()

####################################################################################
# For spdlog
####################################################################################

if(NOT EXISTS ${LIB_DIR}/spdlog)
    find_package(Git REQUIRED)
    execute_process(
            COMMAND ${GIT_EXECUTABLE} clone https://github.com/gabime/spdlog.git ${LIB_DIR}/spdlog
    )
    execute_process(
            COMMAND mkdir -p ${LIB_DIR}/spdlog/build
    )
    execute_process(
            COMMAND cmake .. WORKING_DIRECTORY ${LIB_DIR}/spdlog/build
    )
    execute_process(
            COMMAND make -j WORKING_DIRECTORY ${LIB_DIR}/spdlog/build
    )
endif()

add_subdirectory(${LIB_DIR}/spdlog)

if(NOT TARGET spdlog)
    # Stand-alone build
    find_package(spdlog REQUIRED)
endif()

####################################################################################
# For mjx library
####################################################################################

file(GLOB SOURCE_FILES *.cpp)
file(GLOB HEADER_FILES *.h)
add_library(${PROJECT_NAME} STATIC
        ${SOURCE_FILES} ${HEADER_FILES} ${mj_proto_srcs} ${mj_grpc_srcs})
target_link_libraries(${PROJECT_NAME} ${_REFLECTION} ${_GRPC_GRPCPP} ${_PROTOBUF_LIBPROTOBUF} spdlog::spdlog)

####################################################################################
# For winning hand cache
####################################################################################
add_definitions(-DWIN_CACHE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/cache")

# Generate cache file
add_executable(win_cache_generator win_cache_generator.cpp)
target_link_libraries(win_cache_generator ${PROJECT_NAME})

add_executable(shanten_cache_generator shanten_cache_generator.cpp)
target_link_libraries(shanten_cache_generator ${PROJECT_NAME})
