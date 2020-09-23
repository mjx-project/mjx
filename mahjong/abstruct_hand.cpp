#include "abstruct_hand.h"

#include <string>
#include <utility>

#include "types.h"

namespace mj {

    std::pair<AbstructHand, std::vector<TileType>>
    CreateAbstructHand(const TileTypeCount& count) noexcept {

        // NOTE: そもそもstd::vector<int> hoge(34) で手牌を管理した方がいいかも
        std::vector<int> tile_counts(34);
        std::vector<TileType> tile_types;
        tile_types.reserve(count.size());

        for (auto& [tile_type, n] : count) {
            tile_counts[static_cast<int>(tile_type)] = n;
            tile_types.push_back(tile_type);
        }

        std::string abstruct_hand, hand;

        for (int start : {0, 9, 18}) {
            for (int i = start; i < start + 9; ++i) {
                if (tile_counts[i] > 0) {
                    hand += std::to_string(tile_counts[i]);
                } else if (!hand.empty()) {
                    if (!abstruct_hand.empty()) abstruct_hand += ",";
                    abstruct_hand += hand;
                    hand.clear();
                }
            }
            if (!hand.empty()) {
                if (!abstruct_hand.empty()) abstruct_hand += ",";
                abstruct_hand += hand;
                hand.clear();
            }
        }

        for (int i = 27; i < 34; ++i) {
            if (tile_counts[i] > 0) {
                if (!abstruct_hand.empty()) abstruct_hand += ",";
                abstruct_hand += std::to_string(tile_counts[i]);
            }
        }

        return {abstruct_hand, tile_types};
    }

}
