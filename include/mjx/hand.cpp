#include "mjx/hand.h"

#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/message_differencer.h>

#include <utility>

#include "mjx/shanten_calculator.h"

namespace mjx {
Hand::Hand(mjxproto::Hand proto) : proto_(std::move(proto)) {}

Hand::Hand(const std::string& json) {
  auto status = google::protobuf::util::JsonStringToMessage(json, &proto_);
  assert(status.ok());
}

const mjxproto::Hand& mjx::Hand::proto() const noexcept { return proto_; }

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

std::array<uint8_t, 34> Hand::ClosedTiles() const noexcept {
  std::array<uint8_t, 34> closed_tiles{};
  closed_tiles.fill(0);
  for (auto t : proto_.closed_tiles()) {
    ++closed_tiles[t >> 2];
  }
  return closed_tiles;
}

bool Hand::IsTenpai() const {
  return ShantenCalculator::ShantenNumber(ClosedTiles(), proto_.opens_size()) <=
         0;
}

int Hand::ShantenNumber() const {
  return ShantenCalculator::ShantenNumber(ClosedTiles(), proto_.opens_size());
}

}  // namespace mjx
