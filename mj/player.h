#ifndef MAHJONG_PLAYER_H
#define MAHJONG_PLAYER_H

#include "hand.h"
#include "mj.grpc.pb.h"

namespace mj
{
    struct Player
    {
        Player() = default;
        Player(PlayerId player_id, AbsolutePos position, Hand initial_hand):
                player_id(std::move(player_id)), position(position), hand(std::move(initial_hand))
        {
            assert(hand.stage() == HandStage::kAfterDiscards);
            assert(hand.Size() == 13);
            assert(hand.Opens().empty());
        }

        // Playerクラスにもともとあった変数
        std::bitset<34> machi;    // 上がりの形になるための待ち(役の有無を考慮しない). bitsetで管理する
        std::bitset<34> discards; // 今までに捨てた牌のset. bitsetで管理する
        PlayerId player_id;
        AbsolutePos position;
        Hand hand;

        // Stateクラスから移動してきた変数
        // 他家の打牌でロンを見逃した牌のbitset. フリテンの判定に使用する.
        std::bitset<34> missed_tiles = 0;
        bool is_ippatsu = false;
        bool has_nm = true;
    };
}  // namespace mj

#endif //MAHJONG_PLAYER_H
