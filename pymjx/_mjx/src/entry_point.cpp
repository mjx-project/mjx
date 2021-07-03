#include <mjx/env.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_mjx, m) {
  m.doc() = "";

  py::class_<mjx::env::RLlibMahjongEnv>(m, "RLlibMahjongEnv")
      .def(py::init<>())
      .def("reset", &mjx::env::RLlibMahjongEnv::reset)
      .def("step", &mjx::env::RLlibMahjongEnv::step)
      .def("seed", &mjx::env::RLlibMahjongEnv::seed);
}
