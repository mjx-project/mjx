#include "hand.h"

#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/message_differencer.h>

#include <utility>

namespace mjx {
Hand::Hand(mjxproto::Hand proto) : proto_(std::move(proto)) {}

Hand::Hand(const std::string& json) {
  auto status = google::protobuf::util::JsonStringToMessage(json, &proto_);
  assert(status.ok());
}

const mjxproto::Hand& mjx::Hand::ToProto() const noexcept { return proto_; }

std::string mjx::Hand::ToJson() const noexcept {
  std::string serialized;
  auto status =
      google::protobuf::util::MessageToJsonString(proto_, &serialized);
  assert(status.ok());
  return serialized;
}

bool Hand::operator==(const Hand& other) const noexcept {
  return google::protobuf::util::MessageDifferencer::Equals(proto_,
                                                            other.proto_);
}

bool Hand::operator!=(const Hand& other) const noexcept {
  return !(*this == other);
}
}  // namespace mjx
