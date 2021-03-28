include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${EXTERNALDIR})
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

FetchContent_Declare(
  boostfunctional
  GIT_REPOSITORY https://github.com/boostorg/functional.git
  GIT_TAG        boost-1.75.0
)
FetchContent_GetProperties(boostfunctional)
FetchContent_Populate(boostfunctional)
include_directories(${boostfunctional_SOURCE_DIR}/include)

FetchContent_Declare(
  boostproperty_tree
  GIT_REPOSITORY https://github.com/boostorg/property_tree.git
  GIT_TAG        boost-1.75.0
)
FetchContent_GetProperties(boostproperty_tree)
FetchContent_Populate(boostproperty_tree)
include_directories(${boostproperty_tree_SOURCE_DIR}/include)

FetchContent_Declare(
  boostrandom
  GIT_REPOSITORY https://github.com/boostorg/random
  GIT_TAG        boost-1.75.0
)
FetchContent_GetProperties(boostrandom)
FetchContent_Populate(boostrandom)
include_directories(${boostrandom_SOURCE_DIR}/include)

FetchContent_Declare(
  boostuuid
  GIT_REPOSITORY https://github.com/boostorg/uuid.git
  GIT_TAG        boost-1.75.0
)
FetchContent_GetProperties(boostuuid)
FetchContent_Populate(boostuuid)
include_directories(${boostuuid_SOURCE_DIR}/include)
