include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${EXTERNALDIR})
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

FetchContent_Declare(
    boost_property_tree
    GIT_REPOSITORY https://github.com/boostorg/property_tree.git
    GIT_TAG        boost-1.75.0
)

FetchContent_GetProperties(boost_property_tree)
if(NOT boost_property_tree_POPULATED)
    FetchContent_Populate(boost_property_tree)
    include_directories(${boost_property_tree_SOURCE_DIR}/include)
endif()
