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
        assert(itr_curr_draw_ != draw_end());
        auto drawn_tile = *itr_curr_draw_;
        itr_curr_draw_++;
        return drawn_tile;
    }

    std::array<std::unique_ptr<Hand>, 4> Wall::initial_hands() const {
        std::array<std::unique_ptr<Hand>, 4> hands = {
                std::make_unique<Hand>(initial_hand_begin(0), initial_hand_end(0)),
                std::make_unique<Hand>(initial_hand_begin(1), initial_hand_end(1)),
                std::make_unique<Hand>(initial_hand_begin(2), initial_hand_end(2)),
                std::make_unique<Hand>(initial_hand_begin(3), initial_hand_end(3))
        };
        return hands;
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
}  // namespace mj
