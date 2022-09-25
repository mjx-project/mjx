#include "mjx/event.h"

#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/message_differencer.h>

#include <optional>
#include <utility>

#include "mjx/internal/event.h"

namespace mjx {
Event::Event(mjxproto::Event proto) : proto_(std::move(proto)) {}

Event::Event(const std::string& json) {
  auto status = google::protobuf::util::JsonStringToMessage(json, &proto_);
  assert(status.ok());
}

const mjxproto::Event& mjx::Event::proto() const noexcept { return proto_; }

std::string mjx::Event::ToJson() const noexcept {
  std::string serialized;
  auto status =
      google::protobuf::util::MessageToJsonString(proto_, &serialized);
  assert(status.ok());
  return serialized;
}

bool Event::operator==(const Event& other) const noexcept {
  return google::protobuf::util::MessageDifferencer::Equals(proto_,
                                                            other.proto_);
}

bool Event::operator!=(const Event& other) const noexcept {
  return !(*this == other);
}

int Event::type() const noexcept { return proto_.type(); }

int Event::who() const noexcept { return proto_.who(); }

std::optional<int> Event::tile() const noexcept {
  if (internal::Any(
          type(), {mjxproto::EVENT_TYPE_DISCARD, mjxproto::EVENT_TYPE_TSUMOGIRI,
                   mjxproto::EVENT_TYPE_TSUMO, mjxproto::EVENT_TYPE_RON}))
    return proto_.tile();

  assert(proto_.tile() == 0);
  return std::nullopt;
}

std::optional<int> Event::open() const noexcept {
  if (internal::Any(type(), {mjxproto::EVENT_TYPE_CHI, mjxproto::EVENT_TYPE_PON,
                             mjxproto::EVENT_TYPE_CLOSED_KAN,
                             mjxproto::EVENT_TYPE_OPEN_KAN,
                             mjxproto::EVENT_TYPE_ADDED_KAN}))
    return proto_.open();

  assert(proto_.open() == 0);
  return std::nullopt;
}
}  // namespace mjx
