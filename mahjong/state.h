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
        /*
         * 136 tiles, indexed [0, 135]
         *  - [0, 51] (13*4=52): initial hands of 4 players 配牌
         *  - [52, 121] (70): draws ツモ
         *  - [122, 125] (4): kan draws 嶺上牌
         *  - [126] (1):  dora ドラ
         *  - [127, 130] (4): kan doras カンドラ
         *  - [131] (1): ura dora 裏ドラ
         *  - [132, 135] (4): kan ura doras カンドラ裏
         */
        Wall(std::uint32_t seed = 9999)
        : seed(seed), wall(Tile::CreateAllShuffled(seed)),
        itr_curr_draw(wall.cbegin()), itr_curr_kan_draw(wall.cbegin() + 122),
        itr_dora_begin(wall.cbegin() + 126), itr_ura_dora_begin(wall.cbegin() + 131)
        {}
        const std::uint32_t seed;
        const std::vector<Tile> wall;
        std::vector<Tile>::const_iterator itr_curr_draw;
        std::vector<Tile>::const_iterator itr_curr_kan_draw;
        const std::vector<Tile>::const_iterator itr_dora_begin;
        const std::vector<Tile>::const_iterator itr_ura_dora_begin;
    };

    struct StateInRound
    {
        AbsolutePos dealer;
        Wall wall;
        std::array<River, 4> river;
        std::array<Hand, 4> hand;
    };

    class State
    {
    public:
        State(std::uint32_t seed = 9999);

        void Init(std::uint32_t seed);
        bool IsGameOver();

        // operate or access in-round state
        void InitRound();
        bool IsRoundOver();
        AbsolutePos GetDealerPos();
        AbsolutePos UpdateStateByDraw();
        void UpdateStateByAction(const Action& action);
        Action& UpdateStateByActionCandidates(const std::vector<Action> &action_candidates);
        // operate wall
        Tile Draw();
        void AddNewDora();
        Tile DrawRinshan();

        Observation& GetObservation(AbsolutePos pos) const;
        std::string ToMjlog() const;
    private:
        friend class Environment;
        std::uint32_t seed_;
        Score score_;
        StateInRound state_in_round_;
    };
}  // namespace mj

#endif //MAHJONG_STATE_H
