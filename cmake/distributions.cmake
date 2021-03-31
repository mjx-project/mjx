include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${EXTERNALDIR})
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

FetchContent_Declare(
    distributions
    GIT_REPOSITORY https://github.com/sotetsuk/distributions.git
    GIT_TAG        v0.1.0
)

FetchContent_GetProperties(distributions)
if(NOT distributions_POPULATED)
    FetchContent_Populate(distributions)
    include_directories(${distributions_SOURCE_DIR}/include)
endif()
