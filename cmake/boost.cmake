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
    include_directories(${boost_SOURCE_DIR}/libs/*/include)
endif()
