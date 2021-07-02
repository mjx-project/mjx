#include "observation.h"

#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/message_differencer.h>

#include <utility>

namespace mjx {
Observation::Observation(mjxproto::Observation proto)
    : proto_(std::move(proto)) {}

Observation::Observation(const std::string& json) {
  auto status = google::protobuf::util::JsonStringToMessage(json, &proto_);
  assert(status.ok());
}

const mjxproto::Observation& mjx::Observation::ToProto() const noexcept {
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
}  // namespace mjx
