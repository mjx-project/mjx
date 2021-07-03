#include "mjx/action.h"

#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/message_differencer.h>

#include <utility>

namespace mjx {
Action::Action(mjxproto::Action proto) : proto_(std::move(proto)) {}

Action::Action(const std::string& json) {
  auto status = google::protobuf::util::JsonStringToMessage(json, &proto_);
  assert(status.ok());
}

const mjxproto::Action& mjx::Action::ToProto() const noexcept { return proto_; }

std::string mjx::Action::ToJson() const noexcept {
  std::string serialized;
  auto status =
      google::protobuf::util::MessageToJsonString(proto_, &serialized);
  assert(status.ok());
  return serialized;
}

bool Action::operator==(const Action& other) const noexcept {
  return google::protobuf::util::MessageDifferencer::Equals(proto_,
                                                            other.proto_);
}

bool Action::operator!=(const Action& other) const noexcept {
  return !(*this == other);
}
}  // namespace mjx
