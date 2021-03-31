include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${EXTERNALDIR})
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

FetchContent_Declare(
    boost
    GIT_REPOSITORY https://github.com/boostorg/boost.git
    GIT_TAG        boost-1.75.0
)

FetchContent_GetProperties(boost)
if(NOT boost_POPULATED)
    FetchContent_Populate(boost)
    include_directories(${boost_SOURCE_DIR}/libs/assert/include)
    include_directories(${boost_SOURCE_DIR}/libs/config/include)
    include_directories(${boost_SOURCE_DIR}/libs/core/include)
    include_directories(${boost_SOURCE_DIR}/libs/exception/include)
    include_directories(${boost_SOURCE_DIR}/libs/integer/include)
    include_directories(${boost_SOURCE_DIR}/libs/multi_index/include)
    include_directories(${boost_SOURCE_DIR}/libs/optional/include)
    include_directories(${boost_SOURCE_DIR}/libs/property_tree/include)
    include_directories(${boost_SOURCE_DIR}/libs/random/include)
    include_directories(${boost_SOURCE_DIR}/libs/type_traits/include)
    include_directories(${boost_SOURCE_DIR}/libs/utility/include)
    include_directories(${boost_SOURCE_DIR}/libs/uuid/include)
endif()
