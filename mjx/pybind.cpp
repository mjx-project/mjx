#include <mjx/env.h>
#include <mjx/open.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>

namespace py = pybind11;

// [references]
// https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python
// https://pybind11.readthedocs.io/en/stable/reference.html#c.PYBIND11_OVERRIDE_NAME
class PyAgent : public mjx::Agent {
 public:
  using mjx::Agent::Agent;
  mjx::Action Act(const mjx::Observation &observation) const noexcept override {
    PYBIND11_OVERRIDE_PURE_NAME(mjx::Action, mjx::Agent, "_act", Act,
                                observation);
  }
  std::vector<mjx::Action> ActBatch(const std::vector<mjx::Observation>
                                        &observations) const noexcept override {
    PYBIND11_OVERRIDE_PURE_NAME(std::vector<mjx::Action>, mjx::Agent,
                                "_act_batch", ActBatch, observations);
  }
};

PYBIND11_MODULE(_mjx, m) {
  m.doc() = "";

  py::class_<mjx::Action>(m, "Action")
      .def(py::init<std::string>())
      .def("select_from", &mjx::Action::SelectFrom)
      .def("to_json", &mjx::Action::ToJson)
      .def("to_idx", &mjx::Action::ToIdx)
      .def("type", &mjx::Action::type)
      .def("_open", &mjx::Action::open)
      .def("tile", &mjx::Action::tile)
      .def(py::self == py::self)
      .def(py::self != py::self);

  py::class_<mjx::Event>(m, "Event")
      .def(py::init<std::string>())
      .def("to_json", &mjx::Event::ToJson)
      .def("who", &mjx::Event::who)
      .def("type", &mjx::Event::type)
      .def("_open", &mjx::Event::open)
      .def("tile", &mjx::Event::tile);

  py::class_<mjx::Open>(m, "Open")
      .def_static("event_type", &mjx::Open::EventType)
      .def_static("steal_from", &mjx::Open::From)
      .def_static("at", &mjx::Open::At)
      .def_static("size", &mjx::Open::Size)
      .def_static("tiles", &mjx::Open::Tiles)
      .def_static("tiles_from_hand", &mjx::Open::TilesFromHand)
      .def_static("stolen_tile", &mjx::Open::StolenTile)
      .def_static("last_tile", &mjx::Open::LastTile)
      .def_static("undiscardable_tile_types",
                  &mjx::Open::UndiscardableTileTypes);

  py::class_<mjx::Observation>(m, "Observation")
      .def(py::init<std::string>())
      .def("to_json", &mjx::Observation::ToJson)
      .def("to_features_2d", &mjx::Observation::ToFeatures2D)
      .def("legal_actions", &mjx::Observation::legal_actions)
      .def("events", &mjx::Observation::events)
      .def("draw_history", &mjx::Observation::draw_history)
      .def("who", &mjx::Observation::who)
      .def("dealer", &mjx::Observation::dealer)
      .def("action_mask", &mjx::Observation::action_mask)
      .def("doras", &mjx::Observation::doras)
      .def("kyotaku", &mjx::Observation::kyotaku)
      .def("honba", &mjx::Observation::honba)
      .def("tens", &mjx::Observation::tens)
      .def("round", &mjx::Observation::round)
      .def("curr_hand", &mjx::Observation::curr_hand)
      .def_static("add_legal_actions", &mjx::Observation::AddLegalActions);

  py::class_<mjx::State>(m, "State")
      .def(py::init<std::string>())
      .def("to_json", &mjx::State::ToJson)
      .def("past_decisions", &mjx::State::past_decisions);

  py::class_<mjx::Hand>(m, "Hand")
      .def(py::init<std::string>())
      .def("to_json", &mjx::Hand::ToJson)
      .def("is_tenpai", &mjx::Hand::IsTenpai)
      .def("shanten_number", &mjx::Hand::ShantenNumber)
      .def("effective_draw_types", &mjx::Hand::EffectiveDrawTypes)
      .def("effective_discard_types", &mjx::Hand::EffectiveDiscardTypes)
      .def("closed_tile_types", &mjx::Hand::ClosedTileTypes)
      .def("closed_tiles", &mjx::Hand::ClosedTiles)
      .def("opens", &mjx::Hand::Opens);

  py::class_<mjx::Agent, PyAgent>(m, "Agent")
      .def(py::init<>())
      .def("_act", &mjx::Agent::Act)
      .def("_act_batch", &mjx::Agent::ActBatch);

  py::class_<mjx::RandomDebugAgent, mjx::Agent>(m, "RandomDebugAgent")
      .def(py::init<>());

  py::class_<mjx::RuleBasedAgent, mjx::Agent>(m, "RuleBasedAgent")
      .def(py::init<>());

  py::class_<mjx::GrpcAgent, mjx::Agent>(m, "GrpcAgent")
      .def(py::init<const std::string &>());

  py::class_<mjx::SeedGenerator>(m, "SeedGenerator");

  py::class_<mjx::RandomSeedGenerator, mjx::SeedGenerator>(
      m, "RandomSeedGenerator")
      .def(py::init<std::vector<mjx::PlayerId>>())
      .def("get", &mjx::RandomSeedGenerator::Get);

  py::class_<mjx::DuplicateRandomSeedGenerator, mjx::SeedGenerator>(
      m, "DuplicateRandomSeedGenerator")
      .def(py::init<std::vector<mjx::PlayerId>>())
      .def("get", &mjx::DuplicateRandomSeedGenerator::Get);

  py::class_<mjx::EnvRunner>(m, "EnvRunner")
      .def(py::init<const std::unordered_map<mjx::PlayerId, mjx::Agent *> &,
                    mjx::SeedGenerator *, int, int, int,
                    std::optional<std::string>, std::optional<std::string>>());

  py::class_<mjx::AgentServer>(m, "AgentServer")
      .def(py::init<const mjx::Agent *, const std::string &, int, int, int>());

  py::class_<mjx::MjxEnv>(m, "MjxEnv")
      .def(py::init<std::vector<mjx::PlayerId>>())
      .def("reset", &mjx::MjxEnv::Reset)
      .def("step", &mjx::MjxEnv::Step)
      .def("done", &mjx::MjxEnv::Done)
      .def("rewards", &mjx::MjxEnv::Rewards)
      .def("state", &mjx::MjxEnv::state);
}
