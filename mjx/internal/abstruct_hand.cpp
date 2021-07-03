#include "mjx/internal/abstruct_hand.h"

#include <string>
#include <utility>

#include "mjx/internal/types.h"

namespace mjx::internal {

AbstructHand CreateAbstructHand(const std::vector<int>& tile_counts) noexcept {
  std::string hand;
  bool need_comma = true;

  for (int start : {0, 9, 18}) {
    for (int i = start; i < start + 9; ++i) {
      if (tile_counts[i] > 0) {
        if (need_comma and !hand.empty()) hand += ",";
        hand += std::to_string(tile_counts[i]);
        need_comma = false;
      } else {
        need_comma = true;
      }
    }
    need_comma = true;
  }

  for (int i = 27; i < 34; ++i) {
    if (tile_counts[i] > 0) {
      if (!hand.empty()) hand += ",";
      hand += std::to_string(tile_counts[i]);
    }
  }

  return hand;
}

AbstructHand CreateAbstructHand(const TileTypeCount& count) noexcept {
  std::vector<int> tile_counts(34);
  for (auto& [tile_type, n] : count) {
    tile_counts[static_cast<int>(tile_type)] = n;
  }

  AbstructHand hand = CreateAbstructHand(tile_counts);
  return hand;
}

std::pair<AbstructHand, std::vector<TileType>> CreateAbstructHandWithTileTypes(
    const TileTypeCount& count) noexcept {
  std::vector<int> tile_counts(34);
  std::vector<TileType> tile_types;
  tile_types.reserve(count.size());

  for (auto& [tile_type, n] : count) {
    tile_counts[static_cast<int>(tile_type)] = n;
    tile_types.push_back(tile_type);
  }

  AbstructHand hand = CreateAbstructHand(tile_counts);
  return {hand, tile_types};
}
}  // namespace mjx::internal
