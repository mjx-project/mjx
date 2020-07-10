#include "player.h"

namespace mj
{
    Player::Player(AbsolutePos position, River river, Hand hand):
    position_(position), river_(std::move(river)), hand_(std::move(hand))
    {}

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
