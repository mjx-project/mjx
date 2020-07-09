#ifndef MAHJONG_WIN_INFO_H
#define MAHJONG_WIN_INFO_H

#include "unordered_set"
#include "vector"

#include "tile.h"
#include "open.h"

namespace mj {

    struct WinningInfo {
        const std::unordered_set<Tile, HashTile>& closed_tiles;
        const std::vector<std::unique_ptr<Open>>& opens;
        const std::optional<Tile>& last_tile_added;
        const HandStage stage;
        const bool under_riichi;
        const TileTypeCount closed_tile_types, all_tile_types;
        const bool is_menzen;
        WinningInfo(
                const std::unordered_set<Tile, HashTile>& closed_tiles,
                const std::vector<std::unique_ptr<Open>>& opens,
                const std::optional<Tile>& last_tile_added,
                HandStage stage,
                bool under_riichi,
                TileTypeCount closed_tile_types,
                TileTypeCount all_tile_types,
                bool is_menzen
                ) noexcept ;
    };

} // namespace mj

#endif //MAHJONG_WIN_INFO_H
