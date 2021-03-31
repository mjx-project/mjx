project(mjx)
set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_libs)

####################################################################################
# For Boost
####################################################################################

# include(${CMAKEDIR}/boost.cmake)
include_directories(boost/libs/system/include)
include_directories(boost/libs/multi_array/include)
include_directories(boost/libs/math/include)
include_directories(boost/libs/smart_ptr/include)
include_directories(boost/libs/parameter/include)
include_directories(boost/libs/algorithm/include)
include_directories(boost/libs/any/include)
include_directories(boost/libs/concept_check/include)
include_directories(boost/libs/python/include)
include_directories(boost/libs/tti/include)
include_directories(boost/libs/functional/include)
include_directories(boost/libs/config/include)
include_directories(boost/libs/log/include)
include_directories(boost/libs/interprocess/include)
include_directories(boost/libs/exception/include)
include_directories(boost/libs/foreach/include)
include_directories(boost/libs/spirit/include)
include_directories(boost/libs/io/include)
include_directories(boost/libs/disjoint_sets/include)
include_directories(boost/libs/units/include)
include_directories(boost/libs/preprocessor/include)
include_directories(boost/libs/format/include)
include_directories(boost/libs/xpressive/include)
include_directories(boost/libs/integer/include)
include_directories(boost/libs/thread/include)
include_directories(boost/libs/tokenizer/include)
include_directories(boost/libs/timer/include)
include_directories(boost/libs/regex/include)
include_directories(boost/libs/crc/include)
include_directories(boost/libs/random/include)
include_directories(boost/libs/serialization/include)
include_directories(boost/libs/test/include)
include_directories(boost/libs/date_time/include)
include_directories(boost/libs/logic/include)
include_directories(boost/libs/graph/include)
include_directories(boost/libs/numeric/conversion/include)
include_directories(boost/libs/lambda/include)
include_directories(boost/libs/mpl/include)
include_directories(boost/libs/typeof/include)
include_directories(boost/libs/tuple/include)
include_directories(boost/libs/utility/include)
include_directories(boost/libs/dynamic_bitset/include)
include_directories(boost/libs/assign/include)
include_directories(boost/libs/filesystem/include)
include_directories(boost/libs/function/include)
include_directories(boost/libs/conversion/include)
include_directories(boost/libs/optional/include)
include_directories(boost/libs/property_tree/include)
include_directories(boost/libs/bimap/include)
include_directories(boost/libs/variant/include)

###################################################################################
# For gRPc
####################################################################################
include(${CMAKEDIR}/grpc.cmake)

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
