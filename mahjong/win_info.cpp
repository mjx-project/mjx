#include "win_info.h"

#include <utility>

#include "types.h"

namespace mj {

    WinningInfo::WinningInfo(
            const std::vector<std::unique_ptr<Open>>& opens,
            std::unordered_set<Tile, HashTile> closed_tiles,
            std::optional<Tile> last_tile_added,
            HandStage stage,
            bool under_riichi,
            TileTypeCount closed_tile_types,
            TileTypeCount all_tile_types,
            bool is_menzen
            ) noexcept :
                opens(opens),
                closed_tiles(std::move(closed_tiles)),
                last_tile_added(last_tile_added),
                stage(stage),
                under_riichi(under_riichi),
                closed_tile_types(std::move(closed_tile_types)),
                all_tile_types(std::move(all_tile_types)),
                is_menzen(is_menzen)
                {}

    WinningInfo& WinningInfo::Ron(Tile tile) noexcept {
        // WARNING: closed_tiles との整合性は取れていない. 上がりの判定だけに使うこと.
        ++closed_tile_types[tile.Type()];
        ++all_tile_types[tile.Type()];
        last_tile_added = tile;
        stage = HandStage::kAfterRon;
        return *this;
    }

    WinningInfo& WinningInfo::Discard(Tile tile) noexcept {
        // WARNING: closed_tiles との整合性は取れていない. 上がりの判定だけに使うこと.
        const auto tile_type = tile.Type();
        if (--closed_tile_types[tile_type] == 0) {
            closed_tile_types.erase(tile_type);
        }
        if (--all_tile_types[tile_type] == 0) {
            all_tile_types.erase(tile_type);
        }
        stage = HandStage::kAfterDiscards;
        return *this;
    }

    WinningInfo& WinningInfo::Tsumo(TileType tile_type) noexcept {
        // WARNING: closed_tiles との整合性は取れていない. 上がりの判定だけに使うこと.
        ++closed_tile_types[tile_type];
        ++all_tile_types[tile_type];
        last_tile_added = Tile(tile_type);  // WARNING: 適当にTileを生成している. 既に使用している牌の可能性もある. 空聴のためなのでやむなし.
        stage = HandStage::kAfterTsumo;
        return *this;
    }
}