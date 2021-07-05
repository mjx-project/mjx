#include <mjx/env.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_mjx, m) {
  m.doc() = "";

  py::class_<mjx::Action>(m, "Action")
      .def(py::init<>())
      .def("to_json", &mjx::Action::ToJson)
      .def("idx", &mjx::Action::idx);

  py::class_<mjx::Observation>(m, "Observation")
      .def(py::init<>())
      .def("to_json", &mjx::Observation::ToJson)
      .def("feature", &mjx::Observation::feature)
      .def("legal_actions", &mjx::Observation::legal_actions)
      .def("action_mask", &mjx::Observation::action_mask);

  py::class_<mjx::RLlibMahjongEnv>(m, "RLlibMahjongEnv")
      .def(py::init<>())
      .def("reset", &mjx::RLlibMahjongEnv::reset)
      .def("step", &mjx::RLlibMahjongEnv::step)
      .def("seed", &mjx::RLlibMahjongEnv::seed);
}
