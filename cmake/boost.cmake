include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${EXTERNALDIR})
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

FetchContent_Declare(
    boost
    GIT_REPOSITORY https://github.com/boostorg/boost.git
    GIT_TAG        boost-1.71.0
)

FetchContent_GetProperties(boost)
if(NOT boost_POPULATED)
    FetchContent_Populate(boost)
    include_directories(${boost_SOURCE_DIR}/libs/system/include)
    include_directories(${boost_SOURCE_DIR}/libs/multi_array/include)
    include_directories(${boost_SOURCE_DIR}/libs/math/include)
    include_directories(${boost_SOURCE_DIR}/libs/smart_ptr/include)
    include_directories(${boost_SOURCE_DIR}/libs/parameter/include)
    include_directories(${boost_SOURCE_DIR}/libs/algorithm/include)
    include_directories(${boost_SOURCE_DIR}/libs/any/include)
    include_directories(${boost_SOURCE_DIR}/libs/concept_check/include)
    include_directories(${boost_SOURCE_DIR}/libs/python/include)
    include_directories(${boost_SOURCE_DIR}/libs/tti/include)
    include_directories(${boost_SOURCE_DIR}/libs/functional/include)
    include_directories(${boost_SOURCE_DIR}/libs/config/include)
    include_directories(${boost_SOURCE_DIR}/libs/log/include)
    include_directories(${boost_SOURCE_DIR}/libs/interprocess/include)
    include_directories(${boost_SOURCE_DIR}/libs/exception/include)
    include_directories(${boost_SOURCE_DIR}/libs/foreach/include)
    include_directories(${boost_SOURCE_DIR}/libs/spirit/include)
    include_directories(${boost_SOURCE_DIR}/libs/io/include)
    include_directories(${boost_SOURCE_DIR}/libs/disjoint_sets/include)
    include_directories(${boost_SOURCE_DIR}/libs/units/include)
    include_directories(${boost_SOURCE_DIR}/libs/preprocessor/include)
    include_directories(${boost_SOURCE_DIR}/libs/format/include)
    include_directories(${boost_SOURCE_DIR}/libs/xpressive/include)
    include_directories(${boost_SOURCE_DIR}/libs/integer/include)
    include_directories(${boost_SOURCE_DIR}/libs/thread/include)
    include_directories(${boost_SOURCE_DIR}/libs/tokenizer/include)
    include_directories(${boost_SOURCE_DIR}/libs/timer/include)
    include_directories(${boost_SOURCE_DIR}/libs/regex/include)
    include_directories(${boost_SOURCE_DIR}/libs/crc/include)
    include_directories(${boost_SOURCE_DIR}/libs/random/include)
    include_directories(${boost_SOURCE_DIR}/libs/serialization/include)
    include_directories(${boost_SOURCE_DIR}/libs/test/include)
    include_directories(${boost_SOURCE_DIR}/libs/date_time/include)
    include_directories(${boost_SOURCE_DIR}/libs/logic/include)
    include_directories(${boost_SOURCE_DIR}/libs/graph/include)
    include_directories(${boost_SOURCE_DIR}/libs/numeric/conversion/include)
    include_directories(${boost_SOURCE_DIR}/libs/lambda/include)
    include_directories(${boost_SOURCE_DIR}/libs/mpl/include)
    include_directories(${boost_SOURCE_DIR}/libs/typeof/include)
    include_directories(${boost_SOURCE_DIR}/libs/tuple/include)
    include_directories(${boost_SOURCE_DIR}/libs/utility/include)
    include_directories(${boost_SOURCE_DIR}/libs/dynamic_bitset/include)
    include_directories(${boost_SOURCE_DIR}/libs/assign/include)
    include_directories(${boost_SOURCE_DIR}/libs/filesystem/include)
    include_directories(${boost_SOURCE_DIR}/libs/function/include)
    include_directories(${boost_SOURCE_DIR}/libs/conversion/include)
    include_directories(${boost_SOURCE_DIR}/libs/optional/include)
    include_directories(${boost_SOURCE_DIR}/libs/property_tree/include)
    include_directories(${boost_SOURCE_DIR}/libs/bimap/include)
    include_directories(${boost_SOURCE_DIR}/libs/variant/include)
    include_directories(${boost_SOURCE_DIR}/libs/array/include)
    include_directories(${boost_SOURCE_DIR}/libs/iostreams/include)
    include_directories(${boost_SOURCE_DIR}/libs/multi_index/include)
    include_directories(${boost_SOURCE_DIR}/libs/ptr_container/include)
    include_directories(${boost_SOURCE_DIR}/libs/statechart/include)
    include_directories(${boost_SOURCE_DIR}/libs/static_assert/include)
    include_directories(${boost_SOURCE_DIR}/libs/range/include)
    include_directories(${boost_SOURCE_DIR}/libs/rational/include)
    include_directories(${boost_SOURCE_DIR}/libs/iterator/include)
    include_directories(${boost_SOURCE_DIR}/libs/graph_parallel/include)
    include_directories(${boost_SOURCE_DIR}/libs/property_map/include)
    include_directories(${boost_SOURCE_DIR}/libs/program_options/include)
    include_directories(${boost_SOURCE_DIR}/libs/detail/include)
    include_directories(${boost_SOURCE_DIR}/libs/numeric/interval/include)
    include_directories(${boost_SOURCE_DIR}/libs/numeric/ublas/include)
    include_directories(${boost_SOURCE_DIR}/libs/wave/include)
    include_directories(${boost_SOURCE_DIR}/libs/type_traits/include)
    include_directories(${boost_SOURCE_DIR}/libs/compatibility/include)
    include_directories(${boost_SOURCE_DIR}/libs/bind/include)
    include_directories(${boost_SOURCE_DIR}/libs/pool/include)
    include_directories(${boost_SOURCE_DIR}/libs/proto/include)
    include_directories(${boost_SOURCE_DIR}/libs/fusion/include)
    include_directories(${boost_SOURCE_DIR}/libs/function_types/include)
    include_directories(${boost_SOURCE_DIR}/libs/gil/include)
    include_directories(${boost_SOURCE_DIR}/libs/intrusive/include)
    include_directories(${boost_SOURCE_DIR}/libs/asio/include)
    include_directories(${boost_SOURCE_DIR}/libs/uuid/include)
    include_directories(${boost_SOURCE_DIR}/libs/circular_buffer/include)
    include_directories(${boost_SOURCE_DIR}/libs/mpi/include)
    include_directories(${boost_SOURCE_DIR}/libs/unordered/include)
    include_directories(${boost_SOURCE_DIR}/libs/signals2/include)
    include_directories(${boost_SOURCE_DIR}/libs/accumulators/include)
    include_directories(${boost_SOURCE_DIR}/libs/atomic/include)
    include_directories(${boost_SOURCE_DIR}/libs/scope_exit/include)
    include_directories(${boost_SOURCE_DIR}/libs/flyweight/include)
    include_directories(${boost_SOURCE_DIR}/libs/icl/include)
    include_directories(${boost_SOURCE_DIR}/libs/predef/include)
    include_directories(${boost_SOURCE_DIR}/libs/chrono/include)
    include_directories(${boost_SOURCE_DIR}/libs/polygon/include)
    include_directories(${boost_SOURCE_DIR}/libs/msm/include)
    include_directories(${boost_SOURCE_DIR}/libs/heap/include)
    include_directories(${boost_SOURCE_DIR}/libs/coroutine/include)
    include_directories(${boost_SOURCE_DIR}/libs/coroutine2/include)
    include_directories(${boost_SOURCE_DIR}/libs/ratio/include)
    include_directories(${boost_SOURCE_DIR}/libs/numeric/odeint/include)
    include_directories(${boost_SOURCE_DIR}/libs/geometry/include)
    include_directories(${boost_SOURCE_DIR}/libs/phoenix/include)
    include_directories(${boost_SOURCE_DIR}/libs/move/include)
    include_directories(${boost_SOURCE_DIR}/libs/locale/include)
    include_directories(${boost_SOURCE_DIR}/libs/container/include)
    include_directories(${boost_SOURCE_DIR}/libs/local_function/include)
    include_directories(${boost_SOURCE_DIR}/libs/context/include)
    include_directories(${boost_SOURCE_DIR}/libs/type_erasure/include)
    include_directories(${boost_SOURCE_DIR}/libs/multiprecision/include)
    include_directories(${boost_SOURCE_DIR}/libs/lockfree/include)
    include_directories(${boost_SOURCE_DIR}/libs/assert/include)
    include_directories(${boost_SOURCE_DIR}/libs/align/include)
    include_directories(${boost_SOURCE_DIR}/libs/type_index/include)
    include_directories(${boost_SOURCE_DIR}/libs/core/include)
    include_directories(${boost_SOURCE_DIR}/libs/throw_exception/include)
    include_directories(${boost_SOURCE_DIR}/libs/winapi/include)
    include_directories(${boost_SOURCE_DIR}/libs/lexical_cast/include)
    include_directories(${boost_SOURCE_DIR}/libs/sort/include)
    include_directories(${boost_SOURCE_DIR}/libs/convert/include)
    include_directories(${boost_SOURCE_DIR}/libs/endian/include)
    include_directories(${boost_SOURCE_DIR}/libs/vmd/include)
    include_directories(${boost_SOURCE_DIR}/libs/dll/include)
    include_directories(${boost_SOURCE_DIR}/libs/compute/include)
    include_directories(${boost_SOURCE_DIR}/libs/hana/include)
    include_directories(${boost_SOURCE_DIR}/libs/metaparse/include)
    include_directories(${boost_SOURCE_DIR}/libs/qvm/include)
    include_directories(${boost_SOURCE_DIR}/libs/fiber/include)
    include_directories(${boost_SOURCE_DIR}/libs/process/include)
    include_directories(${boost_SOURCE_DIR}/libs/stacktrace/include)
    include_directories(${boost_SOURCE_DIR}/libs/poly_collection/include)
    include_directories(${boost_SOURCE_DIR}/libs/beast/include)
    include_directories(${boost_SOURCE_DIR}/libs/mp11/include)
    include_directories(${boost_SOURCE_DIR}/libs/callable_traits/include)
    include_directories(${boost_SOURCE_DIR}/libs/contract/include)
    include_directories(${boost_SOURCE_DIR}/libs/container_hash/include)
    include_directories(${boost_SOURCE_DIR}/libs/hof/include)
    include_directories(${boost_SOURCE_DIR}/libs/yap/include)
    include_directories(${boost_SOURCE_DIR}/libs/safe_numerics/include)
    include_directories(${boost_SOURCE_DIR}/libs/parameter_python/include)
    include_directories(${boost_SOURCE_DIR}/libs/headers/include)
    include_directories(${boost_SOURCE_DIR}/libs/outcome/include)
    include_directories(${boost_SOURCE_DIR}/libs/histogram/include)
    include_directories(${boost_SOURCE_DIR}/libs/variant2/include)
endif()
