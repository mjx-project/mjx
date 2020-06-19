#ifndef MAHJONG_STATE_H
#define MAHJONG_STATE_H

#include <string>
#include <array>
#include <vector>
#include "consts.h"
#include "tile.h"
#include "observation.h"
#include "action.h"
#include "hand.h"
#include "river.h"

namespace mj
{
    struct Score
    {
    public:
        Score();
        std::uint8_t round;  // 局
        std::uint8_t honba;  // 本場
        std::uint8_t riichi;  // リー棒
        std::array<std::int16_t, 4> ten;  // 点 250 start
    };

    class Wall
    {
    public:
        Wall(std::uint32_t seed);
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

    class RoundState
    {
    public:
        RoundState(std::uint32_t round_seed);
        bool IsRoundOver();
        AbsolutePos GetDealer();
        void UpdateStateByDraw(AbsolutePos drawer_pos);
        void UpdateStateByAction(std::unique_ptr<Action>);
        void UpdateStateByKanDora();
        void UpdateStateByKanDraw(AbsolutePos drawer_pos);
        void UpdateStateByRyukyoku();
        std::unique_ptr<Action> UpdateStateByStealActionCandidates(const std::vector<std::unique_ptr<Action>> &action_candidates);
        void UpdateStateByFourKanByDifferentPlayers();
        bool CanSteal(AbsolutePos stealer_pos);
        bool CanRon(AbsolutePos winner_pos);
        bool HasFourKanByDifferentPlayers();
    private:
        Wall wall_state_;
        std::array<River, 4> river_states_;
        std::array<Hand, 4> hand_state_;

        bool HasNoDrawTileLeft();
    };

    class State
    {
    public:
        State(std::uint32_t seed);
        bool IsMatchOver();
       std::unique_ptr<Observation> GetObservation(AbsolutePos pos);
        std::string ToMjlog();
    private:
        Score score_;
        RoundState round_state_;
    };
}  // namespace mj

#endif //MAHJONG_STATE_H
