project(mjx)

####################################################################################
# Options
####################################################################################

# set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_libs)
option(USE_SYSTEM_BOOST "Use Boost previously installed." ON)
option(USE_SYSTEM_GRPC "Use gRPC previously installed." ON)

####################################################################################
# For Boost
####################################################################################

# We use boost as header-only
if (USE_SYSTEM_BOOST)
    message(STATUS "USE_SYSTEM_BOOST=ON")
    find_package(Boost REQUIRED)
else()
    message(STATUS "USE_SYSTEM_BOOST=OFF")
    include(${CMAKEDIR}/downloaded_boost.cmake)
endif()
message(STATUS "Include Boost_INCLUDE_DIRS=${Boost_INCLUDE_DIRS}")

###################################################################################
# For gRPc
####################################################################################

if (USE_SYSTEM_GRPC)
    message(STATUS "USE_SYSTEM_GRPC=ON")
    include(${CMAKEDIR}/preinstalled_grpc.cmake)
else()
    message(STATUS "USE_SYSTEM_GRPC=OFF")
    include(${CMAKEDIR}/downloaded_grpc.cmake)
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
# For mjx library
####################################################################################

file(GLOB SOURCE_FILES *.cpp)
file(GLOB HEADER_FILES *.h)
add_library(${PROJECT_NAME} STATIC
        ${SOURCE_FILES} ${HEADER_FILES} ${mjx_proto_srcs} ${mjx_grpc_srcs} ${mjx_proto_hdrs} ${mjx_grpc_hdrs})
# target_link_libraries(${PROJECT_NAME} ${_REFLECTION} ${_GRPC_GRPCPP} ${_PROTOBUF_LIBPROTOBUF} spdlog::spdlog)
target_include_directories(${PROJECT_NAME} PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${_REFLECTION} ${_GRPC_GRPCPP} ${_PROTOBUF_LIBPROTOBUF})

####################################################################################
# For winning hand cache
####################################################################################
add_definitions(-DWIN_CACHE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/cache")

# Generate cache file
add_executable(win_cache_generator win_cache_generator.cpp)
target_link_libraries(win_cache_generator ${PROJECT_NAME})

add_executable(shanten_cache_generator shanten_cache_generator.cpp)
target_link_libraries(shanten_cache_generator ${PROJECT_NAME})
