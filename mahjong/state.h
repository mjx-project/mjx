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
    class Score
    {
    public:
        Score();
    private:
        std::uint8_t round_;  // 局
        std::uint8_t honba_;  // 本場
        std::uint8_t riichi_;  // リー棒
        std::array<std::int16_t, 4> ten_;  // 点 250 start
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

    class StateInRound
    {
    public:
        StateInRound(std::uint32_t round_seed);
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
        Wall wall_;
        std::array<River, 4> river_;
        std::array<Hand, 4> hand_;
        bool HasNoDrawTileLeft();
    };

    class State
    {
    public:
        State();
        State(std::uint32_t seed);
        void Init(std::uint32_t seed);
        bool IsGameOver();
        bool IsRoundOver();
        std::unique_ptr<Observation> GetObservation(AbsolutePos pos);
        std::string ToMjlog();
    private:
        std::uint32_t seed_;
        Score score_;
        StateInRound state_in_round_;
    };
}  // namespace mj

#endif //MAHJONG_STATE_H
