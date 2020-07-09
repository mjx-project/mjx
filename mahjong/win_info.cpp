#include "win_info.h"

#include <utility>

#include "types.h"

namespace mj {

    WinningInfo::WinningInfo(
            const std::unordered_set<Tile, HashTile>& closed_tiles,
            const std::vector<std::unique_ptr<Open>>& opens,
            const std::optional<Tile>& last_tile_added,
            HandStage stage,
            bool under_riichi,
            TileTypeCount closed_tile_types,
            TileTypeCount all_tile_types,
            bool is_menzen
            ) noexcept :
                closed_tiles(closed_tiles),
                opens(opens),
                last_tile_added(last_tile_added),
                stage(stage),
                under_riichi(under_riichi),
                closed_tile_types(std::move(closed_tile_types)),
                all_tile_types(std::move(all_tile_types)),
                is_menzen(is_menzen)
                {}
}