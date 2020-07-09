#ifndef MAHJONG_PLAYER_H
#define MAHJONG_PLAYER_H

#include "hand.h"
#include "river.h"

namespace mj
{
    class Player
    {
        AbsolutePos position() const;
        const Hand* hand() const;
        Hand* mutable_hand();
        const River river() const;
        River* mutable_river();
    private:
        AbsolutePos position_;
        std::unique_ptr<River> river_;
        std::unique_ptr<Hand> hand_;
    };
}  // namespace mj

#endif //MAHJONG_PLAYER_H
