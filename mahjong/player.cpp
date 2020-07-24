#include "player.h"

namespace mj
{
    Player::Player(AbsolutePos position, River river, Hand initial_hand):
    position_(position), river_(std::move(river)), hand_(std::move(initial_hand))
    {
        assert(hand_.Stage() == HandStage::kAfterDiscards);
        assert(hand_.Size() == 13);
        assert(hand_.Opens().empty());
        for (auto tile: hand_.ToVector()) {
            init_hand_.add_tiles(tile.Id());
        }
    }

    AbsolutePos Player::position() const {
        return position_;
    }

    const Hand &Player::hand() const {
        return hand_;
    }

    Hand &Player::mutable_hand() {
        return hand_;
    }

    const River &Player::river() const {
        return river_;
    }

    River &Player::mutable_river() {
        return river_;
    }
}  // namespace mj
