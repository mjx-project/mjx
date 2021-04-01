include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${EXTERNALDIR})
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

FetchContent_Declare(
    boost
    GIT_REPOSITORY https://github.com/boostorg/boost.git
    GIT_TAG        boost-1.71.0
)

FetchContent_GetProperties(boost)
if(NOT boost_POPULATED)
    FetchContent_Populate(boost)
endif()

set(Boost_INCLUDE_DIRS
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
