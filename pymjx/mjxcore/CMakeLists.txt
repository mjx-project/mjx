# download pybind11
include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${MJX_EXTERNAL_DIR})
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
fetchcontent_declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.6.2
)
fetchcontent_makeavailable(pybind11)

# build target: mjxcore
pybind11_add_module(mjxcore src/entry_point.cpp)
target_include_directories(mjxcore PRIVATE ${MJX_INCLUDE_DIR})
target_link_libraries(mjxcore PRIVATE mjx)
