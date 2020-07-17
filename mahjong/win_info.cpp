#include "win_info.h"

#include <cassert>
#include <utility>

#include "types.h"

namespace mj {
    WinningStateInfo::WinningStateInfo(
            Wind prevalent_wind,
            bool is_bottom,
            bool is_first_tsumo,
            std::set<TileType> dora,
            std::set<TileType> reversed_dora
    ) noexcept :
            prevalent_wind(prevalent_wind),
            is_bottom(is_bottom),
            is_first_tsumo(is_first_tsumo),
            dora(std::move(dora)),
            reversed_dora(std::move(reversed_dora)) {}

    WinningInfo::WinningInfo(
            const std::vector<std::unique_ptr<Open>>& opens,
            std::unordered_set<Tile, HashTile> closed_tiles,
            std::optional<TileType> last_added_tile_type,
            HandStage stage,
            bool under_riichi,
            TileTypeCount closed_tile_types,
            TileTypeCount all_tile_types,
            bool is_menzen
    ) noexcept :
            opens(opens),
            closed_tiles(std::move(closed_tiles)),
            last_added_tile_type(last_added_tile_type),
            stage(stage),
            under_riichi(under_riichi),
            closed_tile_types(std::move(closed_tile_types)),
            all_tile_types(std::move(all_tile_types)),
            is_menzen(is_menzen),
            seat_wind(Wind::kEast),
            prevalent_wind(Wind::kEast),
            is_bottom(false),
            is_ippatsu(false),
            is_double_riichi(false),
            is_first_tsumo(false),
            is_dealer(false)
    {}

    WinningInfo::WinningInfo(
            const WinningStateInfo& win_state_info,
            const std::vector<std::unique_ptr<Open>>& opens,
            std::unordered_set<Tile, HashTile> closed_tiles,
            std::optional<TileType> last_added_tile_type,
            HandStage stage,
            bool under_riichi,
            TileTypeCount closed_tile_types,
            TileTypeCount all_tile_types,
            bool is_menzen
    ) noexcept :
            opens(opens),
            closed_tiles(std::move(closed_tiles)),
            last_added_tile_type(last_added_tile_type),
            stage(stage),
            under_riichi(under_riichi),
            closed_tile_types(std::move(closed_tile_types)),
            all_tile_types(std::move(all_tile_types)),
            is_menzen(is_menzen),
            seat_wind(Wind::kEast),
            prevalent_wind(win_state_info.prevalent_wind),
            is_bottom(win_state_info.is_bottom),
            is_ippatsu(false),
            is_double_riichi(false),
            is_first_tsumo(win_state_info.is_first_tsumo),
            is_dealer(false),
            dora(win_state_info.dora),
            reversed_dora(win_state_info.reversed_dora)
    {}

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

    WinningInfo& WinningInfo::Dora(std::set<TileType> dora) noexcept {
        this->dora = dora;
        return *this;
    }
    WinningInfo& WinningInfo::ReversedDora(std::set<TileType> reversed_dora) noexcept {
        assert(under_riichi);
        this->reversed_dora = reversed_dora;
        return *this;
    }
}