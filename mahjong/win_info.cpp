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
        closed_tiles.insert(tile);
        ++closed_tile_types[tile.Type()];
        ++all_tile_types[tile.Type()];
        last_tile_added = tile;
        stage = HandStage::kAfterRon;
        return *this;
    }

    WinningInfo& WinningInfo::Discard(TileType tile_type) noexcept {
        int id = -1;
        for (int i = 0; i < 4; ++i) {
            if (closed_tiles.find(Tile(tile_type, i)) != closed_tiles.end()) {
                id = i;
                break;
            }
        }
        assert(id != -1);

        Tile tile(tile_type, id);
        closed_tiles.erase(tile);
        if (--closed_tile_types[tile_type] == 0) {
            closed_tile_types.erase(tile_type);
        }
        if (--all_tile_types[tile_type] == 0) {
            all_tile_types.erase(tile_type);
        }
        last_tile_added = tile;
        stage = HandStage::kAfterTsumo;
        return *this;
    }

    WinningInfo& WinningInfo::Tsumo(TileType tile_type) noexcept {
        int id = -1;
        for (int i = 0; i < 4; ++i) {
            if (closed_tiles.find(Tile(tile_type, i)) != closed_tiles.end()) {
                id = i;
                break;
            }
        }
        assert(id != -1);

        Tile tile(tile_type, id);
        closed_tiles.insert(tile);
        ++closed_tile_types[tile_type];
        ++all_tile_types[tile_type];
        last_tile_added = tile;
        stage = HandStage::kAfterTsumo;
        return *this;
    }
}