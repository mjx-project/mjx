#include "win_info.h"

#include <cassert>
#include <utility>

#include "types.h"

namespace mj {
    WinInfo& WinInfo::Opens(std::vector<Open> opens) noexcept {
        hand.opens = std::move(opens);
        return *this;
    }

    WinInfo& WinInfo::ClosedTiles(std::unordered_set<Tile, HashTile> closed_tiles) noexcept {
        hand.closed_tiles = std::move(closed_tiles);
        return *this;
    }

    WinInfo& WinInfo::ClosedTileTypes(TileTypeCount closed_tile_types) noexcept {
        hand.closed_tile_types = std::move(closed_tile_types);
        return *this;
    }
    WinInfo& WinInfo::AllTileTypes(TileTypeCount all_tile_types) noexcept {
        hand.all_tile_types = std::move(all_tile_types);
        return *this;
    }
    WinInfo& WinInfo::IsMenzen(bool is_menzen) noexcept {
        hand.is_menzen = is_menzen;
        return *this;
    }
    WinInfo& WinInfo::UnderRiichi(bool under_riichi) noexcept {
        hand.under_riichi = under_riichi;
        return *this;
    }

    WinInfo& WinInfo::Ron(Tile tile) noexcept {
        assert(hand.closed_tiles.find(tile) == hand.closed_tiles.end());
        hand.closed_tiles.insert(tile);
        const auto tile_type = tile.Type();
        ++hand.closed_tile_types[tile_type];
        ++hand.all_tile_types[tile_type];
        hand.win_tile = tile;
        hand.stage = HandStage::kAfterRon;
        return *this;
    }

    WinInfo& WinInfo::Discard(Tile tile) noexcept {
        assert(hand.closed_tiles.find(tile) != hand.closed_tiles.end());
        hand.closed_tiles.erase(tile);
        const auto tile_type = tile.Type();
        assert(hand.closed_tile_types.count(tile_type));
        if (--hand.closed_tile_types[tile_type] == 0) {
            hand.closed_tile_types.erase(tile_type);
        }
        assert(hand.all_tile_types.count(tile_type));
        if (--hand.all_tile_types[tile_type] == 0) {
            hand.all_tile_types.erase(tile_type);
        }
        hand.stage = HandStage::kAfterDiscards;
        return *this;
    }

    WinInfo& WinInfo::Tsumo(Tile tile) noexcept {
        ++hand.closed_tile_types[tile.Type()];
        assert(hand.closed_tile_types[tile.Type()] <= 4);
        ++hand.all_tile_types[tile.Type()];
        hand.win_tile = tile;
        hand.stage = HandStage::kAfterTsumo;
        return *this;
    }

    WinInfo& WinInfo::Stage(HandStage stage) noexcept {
        hand.stage = stage;
        return *this;
    }

    WinInfo& WinInfo::Seat(Wind wind) noexcept {
        state.seat_wind = wind;
        return *this;
    }

    WinInfo& WinInfo::Prevalent(Wind wind) noexcept {
        state.prevalent_wind = wind;
        return *this;
    }

    WinInfo& WinInfo::IsBottom(bool is_bottom) noexcept {
        state.is_bottom = is_bottom;
        return *this;
    }

    WinInfo& WinInfo::IsIppatsu(bool is_ippatsu) noexcept {
        assert(hand.under_riichi);
        state.is_ippatsu = is_ippatsu;
        return *this;
    }

    WinInfo& WinInfo::IsDoubleRiichi(bool is_double_riichi) noexcept {
        assert(hand.under_riichi);
        state.is_double_riichi = is_double_riichi;
        return *this;
    }

    WinInfo& WinInfo::IsFirstTsumo(bool is_first_tsumo) noexcept {
        assert(hand.stage == HandStage::kAfterTsumo);
        state.is_first_tsumo = is_first_tsumo;
        return *this;
    }

    WinInfo& WinInfo::IsDealer(bool is_dealer) noexcept {
        state.is_dealer = is_dealer;
        return *this;
    }

    WinInfo& WinInfo::Dora(std::map<TileType,int> dora) noexcept {
        state.dora = std::move(dora);
        return *this;
    }
    WinInfo& WinInfo::ReversedDora(std::map<TileType,int> reversed_dora) noexcept {
        state.reversed_dora = std::move(reversed_dora);
        return *this;
    }
}