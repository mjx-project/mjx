#include "mjx/observation.h"

#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/message_differencer.h>

#include <utility>

#include "mjx/internal/observation.h"

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

std::vector<float> Observation::ToFeature(
    const std::string& version) const noexcept {
  auto obs = internal::Observation(proto_);
  return obs.ToFeature(version);
}

std::vector<Action> Observation::legal_actions() const noexcept {
  std::vector<Action> actions;
  for (const auto& action_proto : proto_.legal_actions()) {
    actions.emplace_back(Action(action_proto));
  }
  return actions;
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
}  // namespace mjx
