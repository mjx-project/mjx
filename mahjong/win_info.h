#ifndef MAHJONG_WIN_INFO_H
#define MAHJONG_WIN_INFO_H

#include "unordered_set"
#include "vector"

#include "tile.h"
#include "open.h"

namespace mj {

    struct WinStateInfo {
        WinStateInfo() noexcept;
        Wind seat_wind;
        Wind prevalent_wind;
        bool is_bottom;
        bool is_first_tsumo;
        TileTypeCount dora;
        TileTypeCount reversed_dora;
        WinStateInfo& SeatWind(Wind seat_wind) noexcept ;
        WinStateInfo& PrevalentWind(Wind prevalent_wind) noexcept ;
        WinStateInfo& IsBottom(bool is_bottom) noexcept ;
        WinStateInfo& IsFirstTsumo(bool is_first_tsumo) noexcept ;
        WinStateInfo& Dora(TileTypeCount dora) noexcept ;
        WinStateInfo& ReversedDora(TileTypeCount reversed_dora) noexcept ;
    };

    struct WinInfo {
        std::vector<Open> opens;
        std::unordered_set<Tile, HashTile> closed_tiles;
        std::optional<TileType> last_added_tile_type;
        HandStage stage = HandStage::kAfterTsumo;    // default: kAfterTsumo
        bool under_riichi = false;
        TileTypeCount closed_tile_types, all_tile_types;
        bool is_menzen = false;

        Wind seat_wind = Wind::kEast, prevalent_wind = Wind::kEast;
        bool is_bottom = false, is_ippatsu = false, is_double_riichi = false, is_first_tsumo = false;
        bool is_dealer = false;     // 親:true, 子:false (default:false)
        std::map<TileType,int> dora, reversed_dora;

        WinInfo& Opens(std::vector<Open> opens) noexcept ;
        WinInfo& ClosedTiles(std::unordered_set<Tile, HashTile> closed_tiles) noexcept ;
        WinInfo& LastAddedTileType(std::optional<TileType> last_added_tile_type) noexcept ;
        WinInfo& ClosedTileTypes(TileTypeCount closed_tile_types) noexcept ;
        WinInfo& AllTileTypes(TileTypeCount all_tile_types) noexcept ;
        WinInfo& IsMenzen(bool is_menzen) noexcept ;
        WinInfo& UnderRiichi(bool under_riichi) noexcept ;

        WinInfo& ApplyStateInfo(WinStateInfo win_state_info) noexcept ;

        WinInfo& Ron(Tile tile) noexcept ;
        WinInfo& Discard(Tile tile) noexcept ;
        WinInfo& Tsumo(TileType tile_type) noexcept ;
        WinInfo& Seat(Wind wind) noexcept ;
        WinInfo& Prevalent(Wind wind) noexcept ;
        WinInfo& Stage(HandStage stage) noexcept ;
        WinInfo& IsBottom(bool is_bottom) noexcept ;
        WinInfo& IsIppatsu(bool is_ippatsu) noexcept ;
        WinInfo& IsDoubleRiichi(bool is_double_riichi) noexcept ;
        WinInfo& IsFirstTsumo(bool is_first_tsumo) noexcept ;
        WinInfo& IsDealer(bool is_dealer) noexcept ;
        WinInfo& Dora(std::map<TileType,int> dora) noexcept ;
        WinInfo& ReversedDora(std::map<TileType,int> reversed_dora) noexcept ;
    };

} // namespace mj

#endif //MAHJONG_WIN_INFO_H
