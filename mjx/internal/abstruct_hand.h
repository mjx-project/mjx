#ifndef MAHJONG_ABSTRUCT_HAND_H
#define MAHJONG_ABSTRUCT_HAND_H

#include <string>
#include <utility>

#include "mjx/internal/types.h"

namespace mjx::internal {

using AbstructHand = std::string;

[[nodiscard]] AbstructHand CreateAbstructHand(
    const std::vector<int>& count) noexcept;
[[nodiscard]] AbstructHand CreateAbstructHand(
    const TileTypeCount& count) noexcept;
[[nodiscard]] std::pair<AbstructHand,
                        std::vector<TileType>>  // E.g. "2222" {14, 15, 16, 17}
CreateAbstructHandWithTileTypes(const TileTypeCount& count) noexcept;

}  // namespace mjx::internal

#endif  // MAHJONG_ABSTRUCT_HAND_H
