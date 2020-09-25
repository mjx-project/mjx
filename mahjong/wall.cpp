#include <cassert>
#include <array>
#include "wall.h"
#include "utils.h"

namespace mj
{
    Wall::Wall(std::uint32_t round, std::uint32_t seed)
            : round_(round), seed_(seed),
              tiles_(Tile::CreateAllShuffled(seed))
    {}


    Wall::Wall(std::uint32_t round, std::vector<Tile> tiles)
            : round_(round), seed_(-1),
              tiles_(std::move(tiles))
    {}

    Tile Wall::Draw() {
        assert(HasDrawLeft());
        assert(abs(num_kan_draw_ - num_kan_dora_) <= 1);
        auto drawn_tile = tiles_[draw_ix_];
        draw_ix_++;
        return drawn_tile;
    }

    std::vector<Tile> Wall::initial_hand_tiles(AbsolutePos pos) const {
        auto pos_ix = ToUType(pos);
        auto ix = ((pos_ix % 4 - round_ % 4 + 4) % 4) * 4;
        std::vector<Tile> tiles;
        tiles.reserve(13);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                tiles.emplace_back(tiles_.at(ix++));
            }
            ix += 12;
        }
        ix = (pos_ix % 4 - round_ % 4 + 4) % 4 + 48;
        tiles.emplace_back(tiles_.at(ix));
        assert(tiles.size() == 13);
        return tiles;
    }

    Tile Wall::KanDraw() {
        assert(abs(num_kan_draw_ - num_kan_dora_) <= 1);
        assert(num_kan_draw_ <= 3);
        auto kan_ixs = std::vector<int>{134, 135, 132, 133};
        auto drawn_tile = tiles_[kan_ixs[num_kan_draw_++]];
        return drawn_tile;
    }

    std::pair<Tile, Tile> Wall::AddKanDora() {
        assert(abs(num_kan_draw_ - num_kan_dora_) <= 1);
        assert(num_kan_dora_ <= 3);
        num_kan_dora_++;
        auto kan_dora_indicator = tiles_[130 - 2 * num_kan_dora_];
        auto ura_kan_dora_indicator = tiles_[131 - 2 * num_kan_dora_];
        assert(kan_dora_indicator == dora_indicators().back());
        assert(ura_kan_dora_indicator == ura_dora_indicators().back());
        return {kan_dora_indicator, ura_kan_dora_indicator};
    }

    bool Wall::HasDrawLeft() const {
        assert(abs(num_kan_draw_ - num_kan_dora_) <= 1);
        return draw_ix_ + num_kan_draw_ < 122;
    }

    std::vector<Tile> Wall::dora_indicators() const {
        assert(abs(num_kan_draw_ - num_kan_dora_) <= 1);
        std::vector<Tile> ret = {tiles_[130]};
        for (int i = 0; i < num_kan_dora_; ++i) ret.emplace_back(tiles_[128 - 2 * i]);
        return ret;
    }

    std::vector<Tile> Wall::ura_dora_indicators() const {
        assert(abs(num_kan_draw_ - num_kan_dora_) <= 1);
        std::vector<Tile> ret = {tiles_[131]};
        for (int i = 0; i < num_kan_dora_; ++i) ret.emplace_back(tiles_[129 - 2 * i]);
        return ret;
    }

    const std::vector<Tile>& Wall::tiles() const {
        return tiles_;
    }

    TileType Wall::IndicatorToDora(Tile dora_indicator) {
        switch (dora_indicator.Type()) {
            case TileType::kM9:
                return TileType::kM1;
            case TileType::kP9:
                return TileType::kP1;
            case TileType::kS9:
                return TileType::kS1;
            case TileType::kNW:
                return TileType::kEW;
            case TileType::kRD:
                return TileType::kWD;
            default:
                return TileType(ToUType(dora_indicator.Type()) + 1);
        }
    }

    TileTypeCount Wall::dora_count() const {
        std::map<TileType, int> counter;
        for (const auto &t: dora_indicators()) counter[Wall::IndicatorToDora(t)]++;
        return counter;
    }

    TileTypeCount Wall::ura_dora_count() const {
        std::map<TileType, int> counter;
        for (const auto &t: ura_dora_indicators()) counter[Wall::IndicatorToDora(t)]++;
        return counter;
    }
}  // namespace mj
