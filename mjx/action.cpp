#include "action.h"
#include <google/protobuf/util/json_util.h>
#include <utility>

namespace mjx
{
mjx::Action::Action(mjxproto::Action proto): proto_(std::move(proto)) {}

Action::Action(const std::string& json) {
  auto status = google::protobuf::util::JsonStringToMessage(json, &proto_);
  assert(status.ok());
}

const mjxproto::Action& mjx::Action::ToProto() const {
  return proto_;
}

std::string mjx::Action::ToJson() const {
  std::string serialized;
  auto status =
      google::protobuf::util::MessageToJsonString(proto_, &serialized);
  assert(status.ok());
  return serialized;
}
}  // namespace
