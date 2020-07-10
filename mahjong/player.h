#ifndef MAHJONG_PLAYER_H
#define MAHJONG_PLAYER_H

#include "hand.h"
#include "river.h"

namespace mj
{
    class Player
    {
    public:
        Player() = default;
        Player(AbsolutePos position, River river, Hand hand);
        [[nodiscard]] AbsolutePos position() const;
        [[nodiscard]] const Hand& hand() const;
        Hand& mutable_hand();
        [[nodiscard]] const River& river() const;
        River& mutable_river();
    private:
        AbsolutePos position_;
        River river_;
        Hand hand_;
    };
}  // namespace mj

#endif //MAHJONG_PLAYER_H
