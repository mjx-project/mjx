#include "mjx/state.h"

#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/message_differencer.h>

#include <utility>

namespace mjx {
mjx::State::State(mjxproto::State proto) : proto_(std::move(proto)) {}

State::State(const std::string& json) {
  auto status = google::protobuf::util::JsonStringToMessage(json, &proto_);
  assert(status.ok());
}

const mjxproto::State& mjx::State::ToProto() const noexcept { return proto_; }

std::string mjx::State::ToJson() const noexcept {
  std::string serialized;
  auto status =
      google::protobuf::util::MessageToJsonString(proto_, &serialized);
  assert(status.ok());
  return serialized;
}

bool State::operator==(const State& other) const noexcept {
  return google::protobuf::util::MessageDifferencer::Equals(proto_,
                                                            other.proto_);
}

bool State::operator!=(const State& other) const noexcept {
  return !(*this == other);
}
}  // namespace mjx
