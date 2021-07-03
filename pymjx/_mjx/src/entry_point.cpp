#include <mjx/env.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_mjx, m) {
  m.doc() = "";

  py::class_<mjx::RLlibMahjongEnv>(m, "RLlibMahjongEnv")
      .def(py::init<>())
      .def("reset", &mjx::RLlibMahjongEnv::reset)
      .def("step", &mjx::RLlibMahjongEnv::step)
      .def("seed", &mjx::RLlibMahjongEnv::seed);
}
