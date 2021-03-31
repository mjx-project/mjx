include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${EXTERNALDIR})
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

FetchContent_Declare(
    distributions
    GIT_REPOSITORY https://github.com/boostorg/random.git
    GIT_TAG        boost-1.75.0
)

FetchContent_GetProperties(distributions)
if(NOT distributions_POPULATED)
    FetchContent_Populate(distributions)
    include_directories(${distributions_SOURCE_DIR}/include)
endif()
