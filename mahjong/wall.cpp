#include <cassert>
#include <array>
#include "wall.h"
#include "utils.h"

namespace mj
{
    Wall::Wall(std::uint32_t seed)
            : seed_(seed),
              tiles_(std::make_unique<std::vector<Tile>>(Tile::CreateAllShuffled(seed))),
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
        auto i = ToUType(pos);
        return Hand(initial_hand_begin(i), initial_hand_end(i));
    }

    std::vector<Tile>::const_iterator Wall::initial_hand_begin(int pos) const {
        return tiles_->cbegin() + (13 * pos);
    }

    std::vector<Tile>::const_iterator Wall::initial_hand_end(int pos) const {
        return tiles_->cbegin() + (13 * (pos + 1));
    }

    std::vector<Tile>::const_iterator Wall::draw_begin() const {
        return tiles_->cbegin() + 52;
    }

    std::vector<Tile>::const_iterator Wall::draw_end() const {
        return tiles_->cbegin() + 122;
    }

    std::vector<Tile>::const_iterator Wall::kan_draw_begin() const {
        return tiles_->cbegin() + 122;
    }

    std::vector<Tile>::const_iterator Wall::kan_draw_end() const {
        return tiles_->cbegin() + 126;
    }

    std::vector<Tile>::const_iterator Wall::dora_begin() const {
        return tiles_->cbegin() + 126;
    }

    std::vector<Tile>::const_iterator Wall::dora_end() const {
        return tiles_->cbegin() + 131;
    }

    std::vector<Tile>::const_iterator Wall::ura_dora_begin() const {
        return tiles_->cbegin() + 131;
    }

    std::vector<Tile>::const_iterator Wall::ura_dora_end() const {
        return tiles_->cbegin() + 136;
    }

    std::string Wall::ToString(bool verbose) const {
        std::string s;
        for (auto it = initial_hand_begin(0); it != initial_hand_end(0); ++it) s += it->ToString(verbose) + ",";
        s.pop_back(); s += "\n";
        for (auto it = initial_hand_begin(1); it != initial_hand_end(1); ++it) s += it->ToString(verbose) + ",";
        s.pop_back(); s += "\n";
        for (auto it = initial_hand_begin(2); it != initial_hand_end(2); ++it) s += it->ToString(verbose) + ",";
        s.pop_back(); s += "\n";
        for (auto it = initial_hand_begin(3); it != initial_hand_end(3); ++it) s += it->ToString(verbose) + ",";
        s.pop_back(); s += "\n";
        for (auto it = draw_begin(); it != draw_end();) {
            for (int i = 0; i < 6; ++i) {
                s += it->ToString(verbose) + ",";
                ++it;
                if (it == draw_end()) break;
            }
            s.pop_back(); s += "\n";
        }
        for (auto it = kan_draw_begin(); it != kan_draw_end(); ++it) s += it->ToString(verbose) + ",";
        s.pop_back(); s += "\n";
        for (auto it = dora_begin(); it != dora_end(); ++it) s += it->ToString(verbose) + ",";
        s.pop_back(); s += "\n";
        for (auto it = ura_dora_begin(); it != ura_dora_end(); ++it) s += it->ToString(verbose) + ",";
        s.pop_back(); s += "\n";
        return s;
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
        std::vector<Tile> ret;
        for (auto it = dora_begin(); it != dora_begin() + num_kan_dora_ + 1; ++it) ret.emplace_back(*it);
        return ret;
    }

    std::vector<Tile> Wall::ura_doras() const {
        assert(abs(num_kan_draw_ - num_kan_dora_) <= 1);
        std::vector<Tile> ret;
        for (auto it = ura_dora_begin(); it != ura_dora_begin() + num_kan_dora_ + 1; ++it) ret.emplace_back(*it);
        return ret;
    }
}  // namespace mj
