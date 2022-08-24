#include "mjx/observation.h"

#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/message_differencer.h>

#include <utility>

#include "mjx/internal/observation.h"
#include "mjx/internal/state.h"
#include "mjx/internal/types.h"

namespace mjx {
Observation::Observation(mjxproto::Observation proto)
    : proto_(std::move(proto)) {}

Observation::Observation(const std::string& json) {
  auto status = google::protobuf::util::JsonStringToMessage(json, &proto_);
  assert(status.ok());
}

const mjxproto::Observation& mjx::Observation::proto() const noexcept {
  return proto_;
}

std::string mjx::Observation::ToJson() const noexcept {
  std::string serialized;
  auto status =
      google::protobuf::util::MessageToJsonString(proto_, &serialized);
  assert(status.ok());
  return serialized;
}

bool Observation::operator==(const Observation& other) const noexcept {
  return google::protobuf::util::MessageDifferencer::Equals(proto_,
                                                            other.proto_);
}

bool Observation::operator!=(const Observation& other) const noexcept {
  return !(*this == other);
}

std::vector<std::vector<int>> Observation::ToFeatures2D(
    const std::string& version) const noexcept {
  auto obs = internal::Observation(proto_);
  assert(version == "mjx-small-v0" || version == "han22-v0");
  if (version == "mjx-small-v0")
    return obs.ToFeaturesSmallV0();
  else if (version == "han22-v0")
    return obs.ToFeaturesHan22V0();
}

std::vector<Action> Observation::legal_actions() const noexcept {
  std::vector<Action> actions;
  for (const auto& action_proto : proto_.legal_actions()) {
    actions.emplace_back(Action(action_proto));
  }
  return actions;
}

int Observation::who() const noexcept { return proto_.who(); }

int Observation::dealer() const noexcept {
  return proto_.public_observation().init_score().round() % 4;
}

std::vector<Event> Observation::events() const noexcept {
  std::vector<Event> events;
  for (const auto& e : proto_.public_observation().events()) {
    events.emplace_back(e);
  }
  return events;
}

std::vector<int> Observation::draw_history() const noexcept {
  std::vector<int> draw;
  for (int t : proto_.private_observation().draw_history()) {
    draw.push_back(t);
  }
  return draw;
}

std::vector<int> Observation::doras() const noexcept {
  std::vector<int> doras;
  for (auto t : proto_.public_observation().dora_indicators()) {
    doras.push_back(
        ToUType(internal::IndicatorToDora(internal::Tile(t).Type())));
  }
  return doras;
}

std::vector<int> Observation::action_mask() const noexcept {
  auto proto_legal_actions = proto_.legal_actions();
  assert(!proto_legal_actions.empty());
  // TODO: remove magic number 181
  auto mask = std::vector<int>(181, 0);
  for (const auto& proto_action : proto_legal_actions) {
    mask[internal::Action::Encode(proto_action)] = 1;
  }
  return mask;
}

Hand Observation::curr_hand() const noexcept {
  return Hand{proto_.private_observation().curr_hand()};
}

int Observation::kyotaku() const noexcept {
  return proto_.public_observation().init_score().riichi();
}

int Observation::honba() const noexcept {
  return proto_.public_observation().init_score().honba();
}

std::vector<int> Observation::tens() const noexcept {
  std::vector<int> tens;
  for (auto t : proto_.public_observation().init_score().tens()) {
    tens.push_back(t);
  }
  for (auto e : proto_.public_observation().events()) {
    if (e.type() == mjxproto::EVENT_TYPE_RIICHI_SCORE_CHANGE) {
      tens[e.who()] -= 1000;
    }
  }
  return tens;
}

int Observation::round() const noexcept {
  return proto_.public_observation().init_score().round();
}

std::string Observation::AddLegalActions(const std::string& obs_json) {
  auto obs = Observation(obs_json);
  mjxproto::Observation obs_proto = obs.proto();
  auto legal_actions =
      mjx::internal::Observation::GenerateLegalActions(obs_proto);
  for (auto a : legal_actions) {
    obs_proto.mutable_legal_actions()->Add(std::move(a));
  }
  return Observation(obs_proto).ToJson();
}
}  // namespace mjx
