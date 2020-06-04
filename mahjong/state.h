#ifndef MAHJONG_STATE_H
#define MAHJONG_STATE_H

#include <string>
#include <array>
#include <vector>
#include "consts.h"
#include "tile.h"
#include "observation.h"
#include "action.h"

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
        void Tsumo(AbsolutePos who);
        void Ron(AbsolutePos who, AbsolutePos from);
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

    class RoundMatchState
    {
    public:
        RoundMatchState(std::uint32_t round_seed);
        bool IsMatchOver();
        AbsolutePos GetDealer();
    };

    class State
    {
    public:
        State(std::uint32_t seed);
        void InitRound();
        bool IsRoundOver();
        AbsolutePos GetDealerPos();
        bool HasNoDrawTileLeft();
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
        std::unique_ptr<Observation> GetObservation(AbsolutePos pos);
        std::string ToMjlog();
    private:
        ScoreState score_state_;
        WallState wall_satets_;
        RiverState river_state_;
        HandState hand_state_;
    };
}
#endif //MAHJONG_STATE_H
