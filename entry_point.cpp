#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mjx/pyenv.h"

namespace py = pybind11;

PYBIND11_MODULE(pyenv, m) {
  m.doc() = "mahjong environment for python";

  // クラス定義
  py::class_<mjx::env::RLlibMahjongPyEnv>(m, "pyenv")
      .def(py::init<>())
      .def("reset", &mjx::env::RLlibMahjongPyEnv::reset)
      .def("step", &mjx::env::RLlibMahjongPyEnv::step)
      .def("seed", &mjx::env::RLlibMahjongPyEnv::seed);
}
