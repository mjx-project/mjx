#include "mjx/hand.h"

#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/message_differencer.h>

#include <utility>

#include "mjx/internal/shanten_calculator.h"
#include "mjx/internal/utils.h"

namespace mjx {
Hand::Hand(mjxproto::Hand proto) : proto_(std::move(proto)) {}

Hand::Hand(const std::string& json) {
  auto status = google::protobuf::util::JsonStringToMessage(json, &proto_);
  assert(status.ok());
}

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

const mjxproto::Hand& mjx::Hand::proto() const noexcept { return proto_; }

std::array<uint8_t, 34> Hand::ClosedTileTypes() const noexcept {
  std::array<uint8_t, 34> closed_tile_types{};
  closed_tile_types.fill(0);
  for (auto t : proto_.closed_tiles()) {
    ++closed_tile_types[t >> 2];
  }
  return closed_tile_types;
}

std::vector<int> Hand::ClosedTiles() const noexcept {
  std::vector<int> closed_tiles{};
  for (auto t : proto_.closed_tiles()) {
    closed_tiles.push_back(t);
  }
  return closed_tiles;
}

std::vector<int> Hand::Opens() const {
  std::vector<int> opens;
  for (auto o : proto_.opens()) {
    opens.push_back(o);
  }
  return opens;
}

bool Hand::IsTenpai() const {
  return mjx::internal::ShantenCalculator::ShantenNumber(
             ClosedTileTypes(), proto_.opens_size()) <= 0;
}

int Hand::ShantenNumber() const {
  return mjx::internal::ShantenCalculator::ShantenNumber(ClosedTileTypes(),
                                                         proto_.opens_size());
}

std::vector<int> Hand::EffectiveDrawTypes() const {
  auto hand = ClosedTileTypes();
  int num_opens = proto_.opens_size();
  int shanten = internal::ShantenCalculator::ShantenNumber(hand, num_opens);
  std::vector<int> effective_tile_types;
  for (int i = 0; i < 34; ++i) {
    if (hand[i] == 4) continue;
    ++hand[i];
    if (shanten > internal::ShantenCalculator::ShantenNumber(hand, num_opens)) {
      effective_tile_types.push_back(i);
    }
    --hand[i];
  }
  return effective_tile_types;
}

std::vector<int> Hand::EffectiveDiscardTypes() const {
  auto hand = ClosedTileTypes();
  int num_opens = proto_.opens_size();
  int shanten = internal::ShantenCalculator::ShantenNumber(hand, num_opens);
  std::vector<int> effective_tile_types;
  for (int i = 0; i < 34; ++i) {
    if (hand[i] == 0) continue;
    --hand[i];
    if (shanten ==
        internal::ShantenCalculator::ShantenNumber(hand, num_opens)) {
      effective_tile_types.push_back(i);
    }
    ++hand[i];
  }
  return effective_tile_types;
}

}  // namespace mjx
