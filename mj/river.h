#ifndef MAHJONG_RIVER_H
#define MAHJONG_RIVER_H

namespace mj
{
    struct DiscardedTile
    {
        DiscardedTile(Tile tile, bool tsumogiri, std::optional<RelativePos> stolen_to = std::nullopt)
        : tile(tile), tsumogiri(tsumogiri), stolen_to(stolen_to) {}
        Tile tile;
        bool tsumogiri;  // whether the discarded tile is the tile just drawn now ツモ切り
        std::optional<RelativePos> stolen_to;
    };

    class River
    {
    public:
        River() = default;
        void Discard(Tile tile, bool tsumogiri) { discarded_tiles_.emplace_back(tile, tsumogiri, std::nullopt); }

        // accessors
        [[nodiscard]] Tile latest_discard() const { return discarded_tiles_.back().tile; }
        [[nodiscard]] std::size_t size() const { return discarded_tiles_.size(); }
    private:
        std::vector<DiscardedTile> discarded_tiles_;
        std::optional<std::vector<DiscardedTile>::iterator> itr_riichi_pos_ = std::nullopt;
    };
}  // namespace mj

#endif //MAHJONG_RIVER_H
