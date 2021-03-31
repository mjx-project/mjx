include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${EXTERNALDIR})
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

FetchContent_Declare(
    stduuid
    GIT_REPOSITORY https://github.com/mariusbancila/stduuid.git
    GIT_TAG        v1.0
)

FetchContent_GetProperties(stduuid)
if(NOT stduuid_POPULATED)
    FetchContent_Populate(stduuid)
    include_directories(${stduuid_SOURCE_DIR}/include)
endif()
