#ifndef MAHJONG_WIN_INFO_H
#define MAHJONG_WIN_INFO_H

#include "unordered_set"
#include "vector"

#include "tile.h"
#include "open.h"

namespace mj {

    struct WinningInfo {
        const std::vector<std::unique_ptr<Open>>& opens;
        std::unordered_set<Tile, HashTile> closed_tiles;
        std::optional<Tile> last_tile_added;
        HandStage stage;
        bool under_riichi;
        TileTypeCount closed_tile_types, all_tile_types;
        bool is_menzen;

        Wind seat_wind, prevalent_wind;
        WinningInfo(
                const std::vector<std::unique_ptr<Open>>& opens,
                std::unordered_set<Tile, HashTile>  closed_tiles,
                std::optional<Tile> last_tile_added,
                HandStage stage,
                bool under_riichi,
                TileTypeCount closed_tile_types,
                TileTypeCount all_tile_types,
                bool is_menzen
                ) noexcept ;

        WinningInfo& Ron(Tile tile) noexcept ;
        WinningInfo& Discard(Tile tile) noexcept ;
        WinningInfo& Tsumo(TileType tile_type) noexcept ;
        WinningInfo& Seat(Wind wind) noexcept ;
        WinningInfo& Prevalent(Wind wind) noexcept ;
    };

} // namespace mj

#endif //MAHJONG_WIN_INFO_H
