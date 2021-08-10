#include <mjx/env.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_mjx, m) {
  m.doc() = "";

  py::class_<mjx::Action>(m, "Action")
      .def(py::init<>())
      .def(py::init<int, const std::vector<mjx::Action>&>())
      .def("to_json", &mjx::Action::ToJson)
      .def("to_idx", &mjx::Action::ToIdx);

  py::class_<mjx::Observation>(m, "Observation")
      .def(py::init<>())
      .def("to_json", &mjx::Observation::ToJson)
      .def("to_feature", &mjx::Observation::ToFeature)
      .def("legal_actions", &mjx::Observation::legal_actions)
      .def("action_mask", &mjx::Observation::action_mask)
      .def("curr_hand", &mjx::Observation::curr_hand);

  py::class_<mjx::Hand>(m, "Hand")
      .def(py::init<>())
      .def("to_json", &mjx::Hand::ToJson)
      .def("is_tenpai", &mjx::Hand::IsTenpai)
      .def("shanten_number", &mjx::Hand::ShantenNumber);

  py::class_<mjx::RLlibMahjongEnv>(m, "RLlibMahjongEnv")
      .def(py::init<>())
      .def("reset", &mjx::RLlibMahjongEnv::Reset)
      .def("step", &mjx::RLlibMahjongEnv::Step)
      .def("seed", &mjx::RLlibMahjongEnv::Seed);

  py::class_<mjx::PettingZooMahjongEnv>(m, "PettingZooMahjongEnv")
      .def(py::init<>())
      .def("last", &mjx::PettingZooMahjongEnv::Last)
      .def("reset", &mjx::PettingZooMahjongEnv::Reset)
      .def("step", &mjx::PettingZooMahjongEnv::Step)
      .def("seed", &mjx::PettingZooMahjongEnv::Seed)
      .def("agents", &mjx::PettingZooMahjongEnv::agents)
      .def("possible_agents", &mjx::PettingZooMahjongEnv::possible_agents)
      .def("agent_selection", &mjx::PettingZooMahjongEnv::agent_selection
}
