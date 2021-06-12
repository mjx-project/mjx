# download pybind11
include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${MJX_EXTERNALDIR})
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
fetchcontent_declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.6.2
)
fetchcontent_makeavailable(pybind11)

# build target: _mjx
pybind11_add_module(_mjx src/entry_point.cpp)
target_include_directories(_mjx PRIVATE ${MJX_INCLUDE_DIR})
target_link_libraries(_mjx PRIVATE mjx)
