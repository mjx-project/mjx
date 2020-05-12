#ifndef MAHJONG_STATE_H
#define MAHJONG_STATE_H

#include <string>
#include <array>
#include <vector>
#include "observation.h"
#include "consts.h"
#include "tile.h"

namespace mj
{
    class ScoreState
    {
    public:
        ScoreState();
        std::uint8_t round;  // 局
        std::uint8_t honba;  // 本場
        std::uint8_t riichi;  // リー棒
        std::array<std::int16_t, 4> ten;  // 点 250 start
        void Tsumo(absolute_pos who);
        void Ron(absolute_pos who, absolute_pos from);
    };

    class WallState
    {
    public:
        WallState(std::uint32_t seed);
        Tile Draw();
        void AddNewDora();
        Tile DrawRinshan();
    private:
        std::uint32_t seed_;
        std::vector<Tile> dora_;
        std::vector<Tile> ura_dora_;
        std::vector<Tile> wall_;
        std::vector<Tile>::iterator curr_tsumo_;
        std::vector<Tile>::iterator curr_kan_dora_;
        std::vector<Tile>::iterator curr_rinshan_;
    };

    class RiverState
    {

    };

    class HandState
    {

    };

    class State
    {
    public:
        std::string ToMjlog();
        Observation& GetObservation(absolute_pos pos);
    private:
        ScoreState score_state_;
        WallState wall_satets_;
        RiverState river_state_;
        HandState hand_state_;
    };
}
#endif //MAHJONG_STATE_H
