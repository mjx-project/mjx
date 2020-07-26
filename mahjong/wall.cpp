#include <cassert>
#include <array>
#include "wall.h"
#include "utils.h"

namespace mj
{
    Wall::Wall(std::uint32_t round, std::uint32_t seed)
            : round_(round), seed_(seed),
              tiles_(Tile::CreateAllShuffled(seed)),
              itr_curr_draw_(draw_begin()), itr_curr_kan_draw_(kan_draw_begin())
    {}


    Wall::Wall(std::uint32_t round, std::vector<Tile> tiles)
            : round_(round), seed_(-1),
              tiles_(std::move(tiles)),
              itr_curr_draw_(draw_begin()), itr_curr_kan_draw_(kan_draw_begin())
    {}

    Tile Wall::Draw() {
        assert(HasDrawLeft());
        assert(abs(num_kan_draw_ - num_kan_dora_) <= 1);
        auto drawn_tile = *itr_curr_draw_;
        itr_curr_draw_++;
        return drawn_tile;
    }

    Hand Wall::initial_hand(AbsolutePos pos) const {
        auto pos_ix = ToUType(pos);
        auto ix = ((pos_ix % 4 - round_ % 4 + 4) % 4) * 4;
        std::vector<Tile> tiles;
        tiles.reserve(13);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                std::cout << ix << std::endl;
                tiles.emplace_back(tiles_.at(ix++));
            }
            ix += 12;
        }
        ix = (pos_ix % 4 - round_ % 4 + 4) % 4 + 48;
        std::cout << ix << std::endl;
        std::cout << " --- " << std::endl;
        tiles.emplace_back(tiles_.at(ix));
        assert(tiles.size() == 13);
        return Hand(tiles);
    }

    std::vector<Tile>::const_iterator Wall::draw_begin() const {
        return tiles_.cbegin() + 52;
    }

    std::vector<Tile>::const_iterator Wall::draw_end() const {
        return tiles_.cbegin() + 122;
    }

    std::vector<Tile>::const_iterator Wall::kan_draw_begin() const {
        return tiles_.cbegin() + 122;
    }

    std::vector<Tile>::const_iterator Wall::kan_draw_end() const {
        return tiles_.cbegin() + 126;
    }

    Tile Wall::KanDraw() {
        assert(abs(num_kan_draw_ - num_kan_dora_) <= 1);
        assert(num_kan_draw_ <= 3);
        auto drawn_tile = *(kan_draw_begin() + num_kan_draw_);
        num_kan_draw_++;
        return drawn_tile;
    }

    void Wall::AddKanDora() {
        assert(abs(num_kan_draw_ - num_kan_dora_) <= 1);
        assert(num_kan_dora_ <= 3);
        num_kan_dora_++;
    }

    bool Wall::HasDrawLeft() {
        assert(abs(num_kan_draw_ - num_kan_dora_) <= 1);
        return itr_curr_draw_ + num_kan_draw_ != draw_end();
    }

    std::vector<Tile> Wall::doras() const {
        assert(abs(num_kan_draw_ - num_kan_dora_) <= 1);
        std::vector<Tile> ret = {tiles_[130]};
        for (int i = 0; i < num_kan_dora_; ++i) ret.emplace_back(tiles_[128 - 2 * i]);
        return ret;
    }

    std::vector<Tile> Wall::ura_doras() const {
        assert(abs(num_kan_draw_ - num_kan_dora_) <= 1);
        std::vector<Tile> ret = {tiles_[131]};
        for (int i = 0; i < num_kan_dora_; ++i) ret.emplace_back(tiles_[129 - 2 * i]);
        return ret;
    }
}  // namespace mj
