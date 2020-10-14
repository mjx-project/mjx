#ifndef MAHJONG_PLAYER_H
#define MAHJONG_PLAYER_H

#include "hand.h"
#include "mj.grpc.pb.h"

namespace mj
{
    struct Player
    {
        Player() = default;
        std::bitset<34> machi_;    // 上がりの形になるための待ち(役の有無を考慮しない). bitsetで管理する
        std::bitset<34> discards_; // 今までに捨てた牌のset. bitsetで管理する
        PlayerId player_id_;
        AbsolutePos position_;
        Hand hand_;

        //Stateクラスから移動してきた変数
        std::bitset<34> missed_tiles_ = 0;
        bool is_ippatsu_ = false;
        bool has_nm_ = true;
    };
}  // namespace mj

#endif //MAHJONG_PLAYER_H
