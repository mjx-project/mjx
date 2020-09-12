#ifndef MAHJONG_ABSTRUCT_HAND_H
#define MAHJONG_ABSTRUCT_HAND_H

#include <string>
#include <utility>

#include "types.h"

namespace mj {

    using AbstructHand = std::string;

    [[nodiscard]] std::pair<AbstructHand, std::vector<TileType>>  // E.g. "2222" {14, 15, 16, 17}
    CreateAbstructHand(const TileTypeCount& count) noexcept ;

} // namespace mj

#endif //MAHJONG_ABSTRUCT_HAND_H
