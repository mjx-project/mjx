#include "mjx/action.h"

#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/message_differencer.h>

#include <optional>
#include <utility>

#include "mjx/internal/action.h"

namespace mjx {
Action::Action(mjxproto::Action proto) : proto_(std::move(proto)) {}

Action::Action(const std::string& json) {
  auto status = google::protobuf::util::JsonStringToMessage(json, &proto_);
  assert(status.ok());
}

const mjxproto::Action& mjx::Action::proto() const noexcept { return proto_; }

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

Action Action::SelectFrom(int action_idx,
                          const std::vector<Action>& legal_actions) {
  for (const auto& legal_action : legal_actions) {
    if (legal_action.ToIdx() == action_idx) {
      return legal_action;
    }
  }
  assert(false);
}

int Action::ToIdx() const noexcept { return internal::Action::Encode(proto_); }

int Action::type() const noexcept { return proto_.type(); }

std::optional<int> Action::tile() const noexcept {
  if (internal::Any(
          type(),
          {mjxproto::ACTION_TYPE_DISCARD, mjxproto::ACTION_TYPE_TSUMOGIRI,
           mjxproto::ACTION_TYPE_TSUMO, mjxproto::ACTION_TYPE_RON}))
    return proto_.tile();

  assert(proto_.tile() == 0);
  return std::nullopt;
}

std::optional<int> Action::open() const noexcept {
  if (internal::Any(
          type(),
          {mjxproto::ACTION_TYPE_CHI, mjxproto::ACTION_TYPE_PON,
           mjxproto::ACTION_TYPE_CLOSED_KAN, mjxproto::ACTION_TYPE_OPEN_KAN,
           mjxproto::ACTION_TYPE_ADDED_KAN}))
    return proto_.open();

  assert(proto_.open() == 0);
  return std::nullopt;
}
}  // namespace mjx
