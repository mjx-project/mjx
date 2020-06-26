#ifndef MAHJONG_RIVER_H
#define MAHJONG_RIVER_H

namespace mj
{
    struct DiscardedTile
    {
        DiscardedTile(Tile tile, bool tsumogiri, std::optional<RelativePos> stolen_to = std::nullopt)
        : tile(tile), tsumogiri(tsumogiri), stolen_to(stolen_to) {}
        Tile tile;
        bool tsumogiri;
        std::optional<RelativePos> stolen_to;
    };

    struct River
    {
        River() = default;
        std::vector<DiscardedTile> discarded_tiles;
        std::optional<std::vector<DiscardedTile>::iterator> itr_riichi_pos = std::nullopt;
    };
}  // namespace mj

#endif //MAHJONG_RIVER_H
