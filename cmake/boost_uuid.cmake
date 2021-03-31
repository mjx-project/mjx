include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${EXTERNALDIR})
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

FetchContent_Declare(
    boost_uuid
    GIT_REPOSITORY https://github.com/boostorg/uuid.git
    GIT_TAG        boost-1.75.0
)

FetchContent_GetProperties(boost_uuid)
if(NOT boost_uuid_POPULATED)
    FetchContent_Populate(boost_uuid)
    include_directories(${boost_uuid_SOURCE_DIR}/include)
endif()
