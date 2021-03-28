include(FetchContent)

set(FETCHCONTENT_BASE_DIR ${EXTERNALDIR})
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

FetchContent_Declare(
  boostfunctional
  GIT_REPOSITORY https://github.com/boostorg/functional.git
  GIT_TAG        boost-1.75.0
)
FetchContent_MakeAvailable(boostfunctional)

FetchContent_Declare(
  boostproperty_tree
  GIT_REPOSITORY https://github.com/boostorg/property_tree.git
  GIT_TAG        boost-1.75.0
)
FetchContent_MakeAvailable(boostproperty_tree)

FetchContent_Declare(
  boostrandom
  GIT_REPOSITORY https://github.com/boostorg/random
  GIT_TAG        boost-1.75.0
)
FetchContent_MakeAvailable(boostrandom)

FetchContent_Declare(
  boostuuid
  GIT_REPOSITORY https://github.com/boostorg/uuid.git
  GIT_TAG        boost-1.75.0
)
FetchContent_MakeAvailable(boostuuid)
