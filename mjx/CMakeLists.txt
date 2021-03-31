project(mjx)
set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_libs)

####################################################################################
# For Boost
####################################################################################

include_directories(boost/libs/accumulators/include)
include_directories(boost/libs/algorithm/include)
include_directories(boost/libs/align/include)
include_directories(boost/libs/any/include)
include_directories(boost/libs/array/include)
include_directories(boost/libs/asio/include)
include_directories(boost/libs/assert/include)
include_directories(boost/libs/assign/include)
include_directories(boost/libs/atomic/include)
include_directories(boost/libs/beast/include)
include_directories(boost/libs/bimap/include)
include_directories(boost/libs/bind/include)
include_directories(boost/libs/callable_traits/include)
include_directories(boost/libs/chrono/include)
include_directories(boost/libs/circular_buffer/include)
include_directories(boost/libs/compatibility/include)
include_directories(boost/libs/compute/include)
include_directories(boost/libs/concept_check/include)
include_directories(boost/libs/config/include)
include_directories(boost/libs/container/include)
include_directories(boost/libs/container_hash/include)
include_directories(boost/libs/context/include)
include_directories(boost/libs/contract/include)
include_directories(boost/libs/conversion/include)
include_directories(boost/libs/convert/include)
include_directories(boost/libs/core/include)
include_directories(boost/libs/coroutine/include)
include_directories(boost/libs/coroutine2/include)
include_directories(boost/libs/crc/include)
include_directories(boost/libs/date_time/include)
include_directories(boost/libs/detail/include)
include_directories(boost/libs/disjoint_sets/include)
include_directories(boost/libs/dll/include)
include_directories(boost/libs/dynamic_bitset/include)
include_directories(boost/libs/endian/include)
include_directories(boost/libs/exception/include)
include_directories(boost/libs/fiber/include)
include_directories(boost/libs/filesystem/include)
include_directories(boost/libs/flyweight/include)
include_directories(boost/libs/foreach/include)
include_directories(boost/libs/format/include)
include_directories(boost/libs/function/include)
include_directories(boost/libs/functional/include)
include_directories(boost/libs/function_types/include)
include_directories(boost/libs/fusion/include)
include_directories(boost/libs/geometry/include)
include_directories(boost/libs/gil/include)
include_directories(boost/libs/graph/include)
include_directories(boost/libs/graph_parallel/include)
include_directories(boost/libs/hana/include)
include_directories(boost/libs/headers/include)
include_directories(boost/libs/heap/include)
include_directories(boost/libs/histogram/include)
include_directories(boost/libs/hof/include)
include_directories(boost/libs/icl/include)
include_directories(boost/libs/integer/include)
include_directories(boost/libs/interprocess/include)
include_directories(boost/libs/intrusive/include)
include_directories(boost/libs/io/include)
include_directories(boost/libs/iostreams/include)
include_directories(boost/libs/iterator/include)
include_directories(boost/libs/lambda/include)
include_directories(boost/libs/lexical_cast/include)
include_directories(boost/libs/locale/include)
include_directories(boost/libs/local_function/include)
include_directories(boost/libs/lockfree/include)
include_directories(boost/libs/log/include)
include_directories(boost/libs/logic/include)
include_directories(boost/libs/math/include)
include_directories(boost/libs/metaparse/include)
include_directories(boost/libs/move/include)
include_directories(boost/libs/mp11/include)
include_directories(boost/libs/mpi/include)
include_directories(boost/libs/mpl/include)
include_directories(boost/libs/msm/include)
include_directories(boost/libs/multi_array/include)
include_directories(boost/libs/multi_index/include)
include_directories(boost/libs/multiprecision/include)
include_directories(boost/libs/numeric/include)
include_directories(boost/libs/optional/include)
include_directories(boost/libs/outcome/include)
include_directories(boost/libs/parameter/include)
include_directories(boost/libs/parameter_python/include)
include_directories(boost/libs/phoenix/include)
include_directories(boost/libs/poly_collection/include)
include_directories(boost/libs/polygon/include)
include_directories(boost/libs/pool/include)
include_directories(boost/libs/predef/include)
include_directories(boost/libs/preprocessor/include)
include_directories(boost/libs/process/include)
include_directories(boost/libs/program_options/include)
include_directories(boost/libs/property_map/include)
include_directories(boost/libs/property_tree/include)
include_directories(boost/libs/proto/include)
include_directories(boost/libs/ptr_container/include)
include_directories(boost/libs/python/include)
include_directories(boost/libs/qvm/include)
include_directories(boost/libs/random/include)
include_directories(boost/libs/range/include)
include_directories(boost/libs/ratio/include)
include_directories(boost/libs/rational/include)
include_directories(boost/libs/regex/include)
include_directories(boost/libs/safe_numerics/include)
include_directories(boost/libs/scope_exit/include)
include_directories(boost/libs/serialization/include)
include_directories(boost/libs/signals2/include)
include_directories(boost/libs/smart_ptr/include)
include_directories(boost/libs/sort/include)
include_directories(boost/libs/spirit/include)
include_directories(boost/libs/stacktrace/include)
include_directories(boost/libs/statechart/include)
include_directories(boost/libs/static_assert/include)
include_directories(boost/libs/system/include)
include_directories(boost/libs/test/include)
include_directories(boost/libs/thread/include)
include_directories(boost/libs/throw_exception/include)
include_directories(boost/libs/timer/include)
include_directories(boost/libs/tokenizer/include)
include_directories(boost/libs/tti/include)
include_directories(boost/libs/tuple/include)
include_directories(boost/libs/type_erasure/include)
include_directories(boost/libs/type_index/include)
include_directories(boost/libs/typeof/include)
include_directories(boost/libs/type_traits/include)
include_directories(boost/libs/units/include)
include_directories(boost/libs/unordered/include)
include_directories(boost/libs/utility/include)
include_directories(boost/libs/uuid/include)
include_directories(boost/libs/variant/include)
include_directories(boost/libs/variant2/include)
include_directories(boost/libs/vmd/include)
include_directories(boost/libs/wave/include)
include_directories(boost/libs/winapi/include)
include_directories(boost/libs/xpressive/include)
include_directories(boost/libs/yap/include)

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
target_include_directories(
    ${PROJECT_NAME}
    ${PROJECT_SOURCE_DIR}/boost/libs/accumulators/include
    ${PROJECT_SOURCE_DIR}/boost/libs/algorithm/include
    ${PROJECT_SOURCE_DIR}/boost/libs/align/include
    ${PROJECT_SOURCE_DIR}/boost/libs/any/include
    ${PROJECT_SOURCE_DIR}/boost/libs/array/include
    ${PROJECT_SOURCE_DIR}/boost/libs/asio/include
    ${PROJECT_SOURCE_DIR}/boost/libs/assert/include
    ${PROJECT_SOURCE_DIR}/boost/libs/assign/include
    ${PROJECT_SOURCE_DIR}/boost/libs/atomic/include
    ${PROJECT_SOURCE_DIR}/boost/libs/beast/include
    ${PROJECT_SOURCE_DIR}/boost/libs/bimap/include
    ${PROJECT_SOURCE_DIR}/boost/libs/bind/include
    ${PROJECT_SOURCE_DIR}/boost/libs/callable_traits/include
    ${PROJECT_SOURCE_DIR}/boost/libs/chrono/include
    ${PROJECT_SOURCE_DIR}/boost/libs/circular_buffer/include
    ${PROJECT_SOURCE_DIR}/boost/libs/compatibility/include
    ${PROJECT_SOURCE_DIR}/boost/libs/compute/include
    ${PROJECT_SOURCE_DIR}/boost/libs/concept_check/include
    ${PROJECT_SOURCE_DIR}/boost/libs/config/include
    ${PROJECT_SOURCE_DIR}/boost/libs/container/include
    ${PROJECT_SOURCE_DIR}/boost/libs/container_hash/include
    ${PROJECT_SOURCE_DIR}/boost/libs/context/include
    ${PROJECT_SOURCE_DIR}/boost/libs/contract/include
    ${PROJECT_SOURCE_DIR}/boost/libs/conversion/include
    ${PROJECT_SOURCE_DIR}/boost/libs/convert/include
    ${PROJECT_SOURCE_DIR}/boost/libs/core/include
    ${PROJECT_SOURCE_DIR}/boost/libs/coroutine/include
    ${PROJECT_SOURCE_DIR}/boost/libs/coroutine2/include
    ${PROJECT_SOURCE_DIR}/boost/libs/crc/include
    ${PROJECT_SOURCE_DIR}/boost/libs/date_time/include
    ${PROJECT_SOURCE_DIR}/boost/libs/detail/include
    ${PROJECT_SOURCE_DIR}/boost/libs/disjoint_sets/include
    ${PROJECT_SOURCE_DIR}/boost/libs/dll/include
    ${PROJECT_SOURCE_DIR}/boost/libs/dynamic_bitset/include
    ${PROJECT_SOURCE_DIR}/boost/libs/endian/include
    ${PROJECT_SOURCE_DIR}/boost/libs/exception/include
    ${PROJECT_SOURCE_DIR}/boost/libs/fiber/include
    ${PROJECT_SOURCE_DIR}/boost/libs/filesystem/include
    ${PROJECT_SOURCE_DIR}/boost/libs/flyweight/include
    ${PROJECT_SOURCE_DIR}/boost/libs/foreach/include
    ${PROJECT_SOURCE_DIR}/boost/libs/format/include
    ${PROJECT_SOURCE_DIR}/boost/libs/function/include
    ${PROJECT_SOURCE_DIR}/boost/libs/functional/include
    ${PROJECT_SOURCE_DIR}/boost/libs/function_types/include
    ${PROJECT_SOURCE_DIR}/boost/libs/fusion/include
    ${PROJECT_SOURCE_DIR}/boost/libs/geometry/include
    ${PROJECT_SOURCE_DIR}/boost/libs/gil/include
    ${PROJECT_SOURCE_DIR}/boost/libs/graph/include
    ${PROJECT_SOURCE_DIR}/boost/libs/graph_parallel/include
    ${PROJECT_SOURCE_DIR}/boost/libs/hana/include
    ${PROJECT_SOURCE_DIR}/boost/libs/headers/include
    ${PROJECT_SOURCE_DIR}/boost/libs/heap/include
    ${PROJECT_SOURCE_DIR}/boost/libs/histogram/include
    ${PROJECT_SOURCE_DIR}/boost/libs/hof/include
    ${PROJECT_SOURCE_DIR}/boost/libs/icl/include
    ${PROJECT_SOURCE_DIR}/boost/libs/integer/include
    ${PROJECT_SOURCE_DIR}/boost/libs/interprocess/include
    ${PROJECT_SOURCE_DIR}/boost/libs/intrusive/include
    ${PROJECT_SOURCE_DIR}/boost/libs/io/include
    ${PROJECT_SOURCE_DIR}/boost/libs/iostreams/include
    ${PROJECT_SOURCE_DIR}/boost/libs/iterator/include
    ${PROJECT_SOURCE_DIR}/boost/libs/lambda/include
    ${PROJECT_SOURCE_DIR}/boost/libs/lexical_cast/include
    ${PROJECT_SOURCE_DIR}/boost/libs/locale/include
    ${PROJECT_SOURCE_DIR}/boost/libs/local_function/include
    ${PROJECT_SOURCE_DIR}/boost/libs/lockfree/include
    ${PROJECT_SOURCE_DIR}/boost/libs/log/include
    ${PROJECT_SOURCE_DIR}/boost/libs/logic/include
    ${PROJECT_SOURCE_DIR}/boost/libs/math/include
    ${PROJECT_SOURCE_DIR}/boost/libs/metaparse/include
    ${PROJECT_SOURCE_DIR}/boost/libs/move/include
    ${PROJECT_SOURCE_DIR}/boost/libs/mp11/include
    ${PROJECT_SOURCE_DIR}/boost/libs/mpi/include
    ${PROJECT_SOURCE_DIR}/boost/libs/mpl/include
    ${PROJECT_SOURCE_DIR}/boost/libs/msm/include
    ${PROJECT_SOURCE_DIR}/boost/libs/multi_array/include
    ${PROJECT_SOURCE_DIR}/boost/libs/multi_index/include
    ${PROJECT_SOURCE_DIR}/boost/libs/multiprecision/include
    ${PROJECT_SOURCE_DIR}/boost/libs/numeric/include
    ${PROJECT_SOURCE_DIR}/boost/libs/optional/include
    ${PROJECT_SOURCE_DIR}/boost/libs/outcome/include
    ${PROJECT_SOURCE_DIR}/boost/libs/parameter/include
    ${PROJECT_SOURCE_DIR}/boost/libs/parameter_python/include
    ${PROJECT_SOURCE_DIR}/boost/libs/phoenix/include
    ${PROJECT_SOURCE_DIR}/boost/libs/poly_collection/include
    ${PROJECT_SOURCE_DIR}/boost/libs/polygon/include
    ${PROJECT_SOURCE_DIR}/boost/libs/pool/include
    ${PROJECT_SOURCE_DIR}/boost/libs/predef/include
    ${PROJECT_SOURCE_DIR}/boost/libs/preprocessor/include
    ${PROJECT_SOURCE_DIR}/boost/libs/process/include
    ${PROJECT_SOURCE_DIR}/boost/libs/program_options/include
    ${PROJECT_SOURCE_DIR}/boost/libs/property_map/include
    ${PROJECT_SOURCE_DIR}/boost/libs/property_tree/include
    ${PROJECT_SOURCE_DIR}/boost/libs/proto/include
    ${PROJECT_SOURCE_DIR}/boost/libs/ptr_container/include
    ${PROJECT_SOURCE_DIR}/boost/libs/python/include
    ${PROJECT_SOURCE_DIR}/boost/libs/qvm/include
    ${PROJECT_SOURCE_DIR}/boost/libs/random/include
    ${PROJECT_SOURCE_DIR}/boost/libs/range/include
    ${PROJECT_SOURCE_DIR}/boost/libs/ratio/include
    ${PROJECT_SOURCE_DIR}/boost/libs/rational/include
    ${PROJECT_SOURCE_DIR}/boost/libs/regex/include
    ${PROJECT_SOURCE_DIR}/boost/libs/safe_numerics/include
    ${PROJECT_SOURCE_DIR}/boost/libs/scope_exit/include
    ${PROJECT_SOURCE_DIR}/boost/libs/serialization/include
    ${PROJECT_SOURCE_DIR}/boost/libs/signals2/include
    ${PROJECT_SOURCE_DIR}/boost/libs/smart_ptr/include
    ${PROJECT_SOURCE_DIR}/boost/libs/sort/include
    ${PROJECT_SOURCE_DIR}/boost/libs/spirit/include
    ${PROJECT_SOURCE_DIR}/boost/libs/stacktrace/include
    ${PROJECT_SOURCE_DIR}/boost/libs/statechart/include
    ${PROJECT_SOURCE_DIR}/boost/libs/static_assert/include
    ${PROJECT_SOURCE_DIR}/boost/libs/system/include
    ${PROJECT_SOURCE_DIR}/boost/libs/test/include
    ${PROJECT_SOURCE_DIR}/boost/libs/thread/include
    ${PROJECT_SOURCE_DIR}/boost/libs/throw_exception/include
    ${PROJECT_SOURCE_DIR}/boost/libs/timer/include
    ${PROJECT_SOURCE_DIR}/boost/libs/tokenizer/include
    ${PROJECT_SOURCE_DIR}/boost/libs/tti/include
    ${PROJECT_SOURCE_DIR}/boost/libs/tuple/include
    ${PROJECT_SOURCE_DIR}/boost/libs/type_erasure/include
    ${PROJECT_SOURCE_DIR}/boost/libs/type_index/include
    ${PROJECT_SOURCE_DIR}/boost/libs/typeof/include
    ${PROJECT_SOURCE_DIR}/boost/libs/type_traits/include
    ${PROJECT_SOURCE_DIR}/boost/libs/units/include
    ${PROJECT_SOURCE_DIR}/boost/libs/unordered/include
    ${PROJECT_SOURCE_DIR}/boost/libs/utility/include
    ${PROJECT_SOURCE_DIR}/boost/libs/uuid/include
    ${PROJECT_SOURCE_DIR}/boost/libs/variant/include
    ${PROJECT_SOURCE_DIR}/boost/libs/variant2/include
    ${PROJECT_SOURCE_DIR}/boost/libs/vmd/include
    ${PROJECT_SOURCE_DIR}/boost/libs/wave/include
    ${PROJECT_SOURCE_DIR}/boost/libs/winapi/include
    ${PROJECT_SOURCE_DIR}/boost/libs/xpressive/include
    ${PROJECT_SOURCE_DIR}/boost/libs/yap/include
    )

####################################################################################
# For winning hand cache
####################################################################################
add_definitions(-DWIN_CACHE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/cache")

# Generate cache file
add_executable(win_cache_generator win_cache_generator.cpp)
target_link_libraries(win_cache_generator ${PROJECT_NAME})

add_executable(shanten_cache_generator shanten_cache_generator.cpp)
target_link_libraries(shanten_cache_generator ${PROJECT_NAME})


