#ifndef MAHJONG_WIN_INFO_H
#define MAHJONG_WIN_INFO_H

#include "unordered_set"
#include "vector"

#include "tile.h"
#include "open.h"

namespace mj {

    struct WinningStateInfo {
        WinningStateInfo() noexcept;
        Wind prevalent_wind;
        bool is_bottom, is_first_tsumo;
        std::set<TileType> dora, reversed_dora;
        WinningStateInfo& PrevalentWind(Wind prevalent_wind) noexcept ;
        WinningStateInfo& IsBottom(bool is_bottom) noexcept ;
        WinningStateInfo& IsFirstTsumo(bool is_first_tsumo) noexcept ;
        WinningStateInfo& Dora(std::set<TileType> dora) noexcept ;
        WinningStateInfo& ReversedDora(std::set<TileType> reversed_dora) noexcept ;
    };

    struct WinningInfo {
        const std::vector<std::unique_ptr<Open>>& opens;
        std::unordered_set<Tile, HashTile> closed_tiles;
        std::optional<TileType> last_added_tile_type;
        HandStage stage;    // default: kAfterTsumo
        bool under_riichi;
        TileTypeCount closed_tile_types, all_tile_types;
        bool is_menzen;

        Wind seat_wind, prevalent_wind;
        bool is_bottom, is_ippatsu, is_double_riichi, is_first_tsumo;
        bool is_dealer;     // 親:true, 子:false (default:false)
        std::set<TileType> dora, reversed_dora;

        WinningInfo(const std::vector<std::unique_ptr<Open>>& opens) noexcept ;
        WinningInfo& ClosedTiles(std::unordered_set<Tile, HashTile> closed_tiles) noexcept ;
        WinningInfo& LastAddedTileType(std::optional<TileType> last_added_tile_type) noexcept ;
        WinningInfo& ClosedTileTypes(TileTypeCount closed_tile_types) noexcept ;
        WinningInfo& AllTileTypes(TileTypeCount all_tile_types) noexcept ;
        WinningInfo& IsMenzen(bool is_menzen) noexcept ;
        WinningInfo& UnderRiichi(bool under_riichi) noexcept ;

        WinningInfo& ApplyStateInfo(WinningStateInfo win_state_info) noexcept ;

        WinningInfo& Ron(Tile tile) noexcept ;
        WinningInfo& Discard(Tile tile) noexcept ;
        WinningInfo& Tsumo(TileType tile_type) noexcept ;
        WinningInfo& Seat(Wind wind) noexcept ;
        WinningInfo& Prevalent(Wind wind) noexcept ;
        WinningInfo& Stage(HandStage stage) noexcept ;
        WinningInfo& IsBottom(bool is_bottom) noexcept ;
        WinningInfo& IsIppatsu(bool is_ippatsu) noexcept ;
        WinningInfo& IsDoubleRiichi(bool is_double_riichi) noexcept ;
        WinningInfo& IsFirstTsumo(bool is_first_tsumo) noexcept ;
        WinningInfo& IsDealer(bool is_dealer) noexcept ;
        WinningInfo& Dora(std::set<TileType> dora) noexcept ;
        WinningInfo& ReversedDora(std::set<TileType> reversed_dora) noexcept ;
    };

} // namespace mj

#endif //MAHJONG_WIN_INFO_H
