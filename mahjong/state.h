#ifndef MAHJONG_STATE_H
#define MAHJONG_STATE_H

#include <string>
#include <array>
#include <vector>
#include <random>
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
        Score(): round(0), honba(0), riichi(0), ten({250, 250, 250, 250}) {}
        std::uint8_t round;  // 局
        std::uint8_t honba;  // 本場
        std::uint8_t riichi;  // リー棒
        std::array<std::int16_t, 4> ten;  // 点 250 start
    };

    struct Wall
    {
        /* 136 tiles
         *  - 0, ..., 52 (13*4): initial hands of 4 players
         *  - 53, ..., 122 (70 tiles): draws
         *  - 123, ..., 126: kan draws (嶺上牌）
         *  - 127: dora
         *  - 128, ..., 131: kan doras
         *  - 132: ura dora
         *  - 133, ..., 136: kan ura doras
         */
        Wall(std::uint32_t seed = 9999)
        : seed(seed), wall(Tile::CreateAllShuffled(seed)), curr_tsumo(wall.cbegin()),
        dora_begin(wall.cend() - 8), ura_dora_begin(dora_begin + 4), num_dora(1),

        {
        }
        const std::uint32_t seed;
        const std::vector<Tile> wall;
        std::vector<Tile>::const_iterator curr_tsumo;
        std::vector<Tile>::const_iterator dora_begin;
        std::vector<Tile>::const_iterator ura_dora_begin;
        std::uint8_t num_dora;
        std::vector<Tile>::iterator curr_kan_dora;
        std::vector<Tile>::iterator curr_rinshan;
    };

    struct StateInRound
    {
        Wall wall;
        std::array<River, 4> river;
        std::array<Hand, 4> hand;
    };

    class State
    {
    public:
        State();
        State(std::uint32_t seed = 9999);

        void Init(std::uint32_t seed);
        bool IsGameOver();

        // operate or access in-round state
        void InitRound();
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
        bool HasNoDrawTileLeft();

        // operate wall
        Tile Draw();
        void AddNewDora();
        Tile DrawRinshan();

        std::unique_ptr<Observation> GetObservation(AbsolutePos pos);
        std::string ToMjlog();
    private:
        friend class Environment;
        std::uint32_t seed_;
        Score score_;
        StateInRound state_in_round_;
    };
}  // namespace mj

#endif //MAHJONG_STATE_H
