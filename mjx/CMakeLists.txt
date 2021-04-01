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

if (USE_SYSTEM_BOOST)
    message(STATUS "USE_SYSTEM_BOOST=ON")
    find_package(Boost)
    if (Boost_FOUND)
        INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
        ADD_DEFINITIONS( "-DHAS_BOOST" )
    endif()
    message(STATUS "Using Boost_INCLUDE_DIRS=${Boost_INCLUDE_DIRS}")
else()
    message(STATUS "USE_SYSTEM_BOOST=OFF")
    include(${CMAKEDIR}/boost.cmake)
endif()

###################################################################################
# For gRPc
####################################################################################

if (USE_SYSTEM_GRPC)
    message(STATUS "USE_SYSTEM_GRPC=ON")
    find_package(Threads REQUIRED)
    set(protobuf_MODULE_COMPATIBLE TRUE)
    find_package(Protobuf CONFIG REQUIRED)
    message(STATUS "Using Protobuf ${Protobuf_VERSION}")
    set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
    set(_REFLECTION gRPC::grpc++_reflection)
    find_program(_PROTOBUF_PROTOC protoc)
    find_package(gRPC CONFIG REQUIRED)
    message(STATUS "Using gRPC ${gRPC_VERSION}")
    set(_GRPC_GRPCPP gRPC::grpc++)
    find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
else()
    message(STATUS "USE_SYSTEM_GRPC=OFF")
    include(${CMAKEDIR}/grpc.cmake)
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
target_link_libraries(${PROJECT_NAME} ${_REFLECTION} ${_GRPC_GRPCPP} ${_PROTOBUF_LIBPROTOBUF})
if (NOT USE_SYSTEM_BOOST)
target_include_directories(
        ${PROJECT_NAME} PUBLIC
        ${boost_SOURCE_DIR}/libs/accumulators/include
        ${boost_SOURCE_DIR}/libs/algorithm/include
        ${boost_SOURCE_DIR}/libs/align/include
        ${boost_SOURCE_DIR}/libs/any/include
        ${boost_SOURCE_DIR}/libs/array/include
        ${boost_SOURCE_DIR}/libs/asio/include
        ${boost_SOURCE_DIR}/libs/assert/include
        ${boost_SOURCE_DIR}/libs/assign/include
        ${boost_SOURCE_DIR}/libs/atomic/include
        ${boost_SOURCE_DIR}/libs/beast/include
        ${boost_SOURCE_DIR}/libs/bimap/include
        ${boost_SOURCE_DIR}/libs/bind/include
        ${boost_SOURCE_DIR}/libs/callable_traits/include
        ${boost_SOURCE_DIR}/libs/chrono/include
        ${boost_SOURCE_DIR}/libs/circular_buffer/include
        ${boost_SOURCE_DIR}/libs/compatibility/include
        ${boost_SOURCE_DIR}/libs/compute/include
        ${boost_SOURCE_DIR}/libs/concept_check/include
        ${boost_SOURCE_DIR}/libs/config/include
        ${boost_SOURCE_DIR}/libs/container/include
        ${boost_SOURCE_DIR}/libs/container_hash/include
        ${boost_SOURCE_DIR}/libs/context/include
        ${boost_SOURCE_DIR}/libs/contract/include
        ${boost_SOURCE_DIR}/libs/conversion/include
        ${boost_SOURCE_DIR}/libs/convert/include
        ${boost_SOURCE_DIR}/libs/core/include
        ${boost_SOURCE_DIR}/libs/coroutine/include
        ${boost_SOURCE_DIR}/libs/coroutine2/include
        ${boost_SOURCE_DIR}/libs/crc/include
        ${boost_SOURCE_DIR}/libs/date_time/include
        ${boost_SOURCE_DIR}/libs/detail/include
        ${boost_SOURCE_DIR}/libs/disjoint_sets/include
        ${boost_SOURCE_DIR}/libs/dll/include
        ${boost_SOURCE_DIR}/libs/dynamic_bitset/include
        ${boost_SOURCE_DIR}/libs/endian/include
        ${boost_SOURCE_DIR}/libs/exception/include
        ${boost_SOURCE_DIR}/libs/fiber/include
        ${boost_SOURCE_DIR}/libs/filesystem/include
        ${boost_SOURCE_DIR}/libs/flyweight/include
        ${boost_SOURCE_DIR}/libs/foreach/include
        ${boost_SOURCE_DIR}/libs/format/include
        ${boost_SOURCE_DIR}/libs/function/include
        ${boost_SOURCE_DIR}/libs/functional/include
        ${boost_SOURCE_DIR}/libs/function_types/include
        ${boost_SOURCE_DIR}/libs/fusion/include
        ${boost_SOURCE_DIR}/libs/geometry/include
        ${boost_SOURCE_DIR}/libs/gil/include
        ${boost_SOURCE_DIR}/libs/graph/include
        ${boost_SOURCE_DIR}/libs/graph_parallel/include
        ${boost_SOURCE_DIR}/libs/hana/include
        ${boost_SOURCE_DIR}/libs/headers/include
        ${boost_SOURCE_DIR}/libs/heap/include
        ${boost_SOURCE_DIR}/libs/histogram/include
        ${boost_SOURCE_DIR}/libs/hof/include
        ${boost_SOURCE_DIR}/libs/icl/include
        ${boost_SOURCE_DIR}/libs/integer/include
        ${boost_SOURCE_DIR}/libs/interprocess/include
        ${boost_SOURCE_DIR}/libs/intrusive/include
        ${boost_SOURCE_DIR}/libs/io/include
        ${boost_SOURCE_DIR}/libs/iostreams/include
        ${boost_SOURCE_DIR}/libs/iterator/include
        ${boost_SOURCE_DIR}/libs/lambda/include
        ${boost_SOURCE_DIR}/libs/lexical_cast/include
        ${boost_SOURCE_DIR}/libs/locale/include
        ${boost_SOURCE_DIR}/libs/local_function/include
        ${boost_SOURCE_DIR}/libs/lockfree/include
        ${boost_SOURCE_DIR}/libs/log/include
        ${boost_SOURCE_DIR}/libs/logic/include
        ${boost_SOURCE_DIR}/libs/math/include
        ${boost_SOURCE_DIR}/libs/metaparse/include
        ${boost_SOURCE_DIR}/libs/move/include
        ${boost_SOURCE_DIR}/libs/mp11/include
        ${boost_SOURCE_DIR}/libs/mpi/include
        ${boost_SOURCE_DIR}/libs/mpl/include
        ${boost_SOURCE_DIR}/libs/msm/include
        ${boost_SOURCE_DIR}/libs/multi_array/include
        ${boost_SOURCE_DIR}/libs/multi_index/include
        ${boost_SOURCE_DIR}/libs/multiprecision/include
        ${boost_SOURCE_DIR}/libs/numeric/include
        ${boost_SOURCE_DIR}/libs/optional/include
        ${boost_SOURCE_DIR}/libs/outcome/include
        ${boost_SOURCE_DIR}/libs/parameter/include
        ${boost_SOURCE_DIR}/libs/parameter_python/include
        ${boost_SOURCE_DIR}/libs/phoenix/include
        ${boost_SOURCE_DIR}/libs/poly_collection/include
        ${boost_SOURCE_DIR}/libs/polygon/include
        ${boost_SOURCE_DIR}/libs/pool/include
        ${boost_SOURCE_DIR}/libs/predef/include
        ${boost_SOURCE_DIR}/libs/preprocessor/include
        ${boost_SOURCE_DIR}/libs/process/include
        ${boost_SOURCE_DIR}/libs/program_options/include
        ${boost_SOURCE_DIR}/libs/property_map/include
        ${boost_SOURCE_DIR}/libs/property_tree/include
        ${boost_SOURCE_DIR}/libs/proto/include
        ${boost_SOURCE_DIR}/libs/ptr_container/include
        ${boost_SOURCE_DIR}/libs/python/include
        ${boost_SOURCE_DIR}/libs/qvm/include
        ${boost_SOURCE_DIR}/libs/random/include
        ${boost_SOURCE_DIR}/libs/range/include
        ${boost_SOURCE_DIR}/libs/ratio/include
        ${boost_SOURCE_DIR}/libs/rational/include
        ${boost_SOURCE_DIR}/libs/regex/include
        ${boost_SOURCE_DIR}/libs/safe_numerics/include
        ${boost_SOURCE_DIR}/libs/scope_exit/include
        ${boost_SOURCE_DIR}/libs/serialization/include
        ${boost_SOURCE_DIR}/libs/signals2/include
        ${boost_SOURCE_DIR}/libs/smart_ptr/include
        ${boost_SOURCE_DIR}/libs/sort/include
        ${boost_SOURCE_DIR}/libs/spirit/include
        ${boost_SOURCE_DIR}/libs/stacktrace/include
        ${boost_SOURCE_DIR}/libs/statechart/include
        ${boost_SOURCE_DIR}/libs/static_assert/include
        ${boost_SOURCE_DIR}/libs/system/include
        ${boost_SOURCE_DIR}/libs/test/include
        ${boost_SOURCE_DIR}/libs/thread/include
        ${boost_SOURCE_DIR}/libs/throw_exception/include
        ${boost_SOURCE_DIR}/libs/timer/include
        ${boost_SOURCE_DIR}/libs/tokenizer/include
        ${boost_SOURCE_DIR}/libs/tti/include
        ${boost_SOURCE_DIR}/libs/tuple/include
        ${boost_SOURCE_DIR}/libs/type_erasure/include
        ${boost_SOURCE_DIR}/libs/type_index/include
        ${boost_SOURCE_DIR}/libs/typeof/include
        ${boost_SOURCE_DIR}/libs/type_traits/include
        ${boost_SOURCE_DIR}/libs/units/include
        ${boost_SOURCE_DIR}/libs/unordered/include
        ${boost_SOURCE_DIR}/libs/utility/include
        ${boost_SOURCE_DIR}/libs/uuid/include
        ${boost_SOURCE_DIR}/libs/variant/include
        ${boost_SOURCE_DIR}/libs/variant2/include
        ${boost_SOURCE_DIR}/libs/vmd/include
        ${boost_SOURCE_DIR}/libs/wave/include
        ${boost_SOURCE_DIR}/libs/winapi/include
        ${boost_SOURCE_DIR}/libs/xpressive/include
        ${boost_SOURCE_DIR}/libs/yap/include
)
endif()

####################################################################################
# For winning hand cache
####################################################################################
add_definitions(-DWIN_CACHE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/cache")

# Generate cache file
add_executable(win_cache_generator win_cache_generator.cpp)
target_link_libraries(win_cache_generator ${PROJECT_NAME})

add_executable(shanten_cache_generator shanten_cache_generator.cpp)
target_link_libraries(shanten_cache_generator ${PROJECT_NAME})
