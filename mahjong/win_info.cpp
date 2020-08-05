#include "win_info.h"

#include <cassert>
#include <utility>

#include "types.h"

namespace mj {
    WinningStateInfo::WinningStateInfo() noexcept :
            prevalent_wind(Wind::kEast), is_bottom(false), is_first_tsumo(false) {}

    WinningStateInfo& WinningStateInfo::PrevalentWind(Wind prevalent_wind) noexcept {
        this->prevalent_wind = prevalent_wind;
        return *this;
    }
    WinningStateInfo& WinningStateInfo::IsBottom(bool is_bottom) noexcept {
        this->is_bottom = is_bottom;
        return *this;
    }
    WinningStateInfo& WinningStateInfo::IsFirstTsumo(bool is_first_tsumo) noexcept {
        this->is_first_tsumo = is_first_tsumo;
        return *this;
    }
    WinningStateInfo& WinningStateInfo::Dora(TileTypeCount dora) noexcept {
        this->dora = dora;
        return *this;
    }
    WinningStateInfo& WinningStateInfo::ReversedDora(TileTypeCount reversed_dora) noexcept {
        this->reversed_dora = reversed_dora;
        return *this;
    }

    WinningStateInfo &WinningStateInfo::SeatWind(Wind seat_wind) noexcept {
        this->seat_wind = seat_wind;
        return *this;
    }

    WinningInfo& WinningInfo::Opens(std::vector<Open> opens) noexcept {
        this->opens = opens;
        return *this;
    }

    WinningInfo& WinningInfo::ClosedTiles(std::unordered_set<Tile, HashTile> closed_tiles) noexcept {
        this->closed_tiles = closed_tiles;
        return *this;
    }
    WinningInfo& WinningInfo::LastAddedTileType(std::optional<TileType> last_added_tile_type) noexcept {
        this->last_added_tile_type = last_added_tile_type;
        return *this;
    }
    WinningInfo& WinningInfo::ClosedTileTypes(TileTypeCount closed_tile_types) noexcept {
        this->closed_tile_types = closed_tile_types;
        return *this;
    }
    WinningInfo& WinningInfo::AllTileTypes(TileTypeCount all_tile_types) noexcept {
        this->all_tile_types = all_tile_types;
        return *this;
    }
    WinningInfo& WinningInfo::IsMenzen(bool is_menzen) noexcept {
        this->is_menzen = is_menzen;
        return *this;
    }
    WinningInfo& WinningInfo::UnderRiichi(bool under_riichi) noexcept {
        this->under_riichi = under_riichi;
        return *this;
    }

    WinningInfo& WinningInfo::ApplyStateInfo(WinningStateInfo win_state_info) noexcept {
        this->seat_wind = win_state_info.seat_wind;
        this->prevalent_wind = win_state_info.prevalent_wind;
        this->is_bottom = win_state_info.is_bottom;
        this->is_first_tsumo = win_state_info.is_first_tsumo;
        this->dora = win_state_info.dora;
        this->reversed_dora = win_state_info.reversed_dora;
        return *this;
    }

    WinningInfo& WinningInfo::Ron(Tile tile) noexcept {
        assert(closed_tiles.find(tile) == closed_tiles.end());
        closed_tiles.insert(tile);
        const auto tile_type = tile.Type();
        ++closed_tile_types[tile_type];
        ++all_tile_types[tile_type];
        last_added_tile_type = tile_type;
        stage = HandStage::kAfterRon;
        return *this;
    }

    WinningInfo& WinningInfo::Discard(Tile tile) noexcept {
        assert(closed_tiles.find(tile) != closed_tiles.end());
        closed_tiles.erase(tile);
        const auto tile_type = tile.Type();
        assert(closed_tile_types.count(tile_type));
        if (--closed_tile_types[tile_type] == 0) {
            closed_tile_types.erase(tile_type);
        }
        assert(all_tile_types.count(tile_type));
        if (--all_tile_types[tile_type] == 0) {
            all_tile_types.erase(tile_type);
        }
        stage = HandStage::kAfterDiscards;
        return *this;
    }

    WinningInfo& WinningInfo::Tsumo(TileType tile_type) noexcept {
        // WARNING: closed_tiles は更新しない.
        //          理由: このメソッドは[仮にこの種類の牌をツモしたときに上がれるか?]を判定するものである.
        //                そのためTile ではなくTileType を引数にとる.
        //                Tileを特定しないと赤ドラの判定ができないが、赤ドラは他に役がなければ上がれないため,
        //                上がれるかどうかの判定には不要である.
        ++closed_tile_types[tile_type];
        assert(closed_tile_types[tile_type] <= 4);
        ++all_tile_types[tile_type];
        last_added_tile_type = tile_type;
        stage = HandStage::kAfterTsumo;
        return *this;
    }

    WinningInfo& WinningInfo::Seat(Wind wind) noexcept {
        seat_wind = wind;
        return *this;
    }

    WinningInfo& WinningInfo::Prevalent(Wind wind) noexcept {
        prevalent_wind = wind;
        return *this;
    }

    WinningInfo& WinningInfo::Stage(HandStage stage) noexcept {
        this->stage = stage;
        return *this;
    }

    WinningInfo& WinningInfo::IsBottom(bool is_bottom) noexcept {
        this->is_bottom = is_bottom;
        return *this;
    }

    WinningInfo& WinningInfo::IsIppatsu(bool is_ippatsu) noexcept {
        assert(under_riichi);
        this->is_ippatsu = is_ippatsu;
        return *this;
    }

    WinningInfo& WinningInfo::IsDoubleRiichi(bool is_double_riichi) noexcept {
        assert(under_riichi);
        this->is_double_riichi = is_double_riichi;
        return *this;
    }

    WinningInfo& WinningInfo::IsFirstTsumo(bool is_first_tsumo) noexcept {
        assert(stage == HandStage::kAfterTsumo);
        this->is_first_tsumo = is_first_tsumo;
        return *this;
    }

    WinningInfo& WinningInfo::IsDealer(bool is_dealer) noexcept {
        this->is_dealer = is_dealer;
        return *this;
    }

    WinningInfo& WinningInfo::Dora(std::map<TileType,int> dora) noexcept {
        this->dora = dora;
        return *this;
    }
    WinningInfo& WinningInfo::ReversedDora(std::map<TileType,int> reversed_dora) noexcept {
        assert(under_riichi);
        this->reversed_dora = reversed_dora;
        return *this;
    }
}