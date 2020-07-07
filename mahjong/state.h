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
        : seed(seed),
          tiles(std::make_unique<std::vector<Tile>>(Tile::CreateAllShuffled(seed))),
          itr_curr_draw(tiles->cbegin() + 52), itr_draw_end(tiles->cbegin() + 122),
          itr_curr_kan_draw(tiles->cbegin() + 122), itr_kan_draw_end(tiles->cbegin() + 130),
          itr_dora_begin(tiles->cbegin() + 126), itr_ura_dora_begin(tiles->cbegin() + 131)
        {}
        std::uint32_t seed;
        std::unique_ptr<std::vector<Tile>> tiles;
        std::vector<Tile>::const_iterator itr_curr_draw;
        std::vector<Tile>::const_iterator itr_draw_end;
        std::vector<Tile>::const_iterator itr_curr_kan_draw;
        std::vector<Tile>::const_iterator itr_kan_draw_end;
        std::vector<Tile>::const_iterator itr_dora_begin;
        std::vector<Tile>::const_iterator itr_ura_dora_begin;
    };

    class State
    {
    public:
        explicit State(std::uint32_t seed = 9999);
        bool IsGameOver();

        // operate or access in-round state
        void InitRound();
        bool IsRoundOver();
        AbsolutePos UpdateStateByDraw();
        void UpdateStateByAction(const Action& action);
        Action& UpdateStateByActionCandidates(const std::vector<Action> &action_candidates);
        // operate wall
        Tile Draw();
        void AddNewDora();
        Tile DrawRinshan();

        // accessors
        [[nodiscard]] const Observation * observation(AbsolutePos who) const;
        Observation * mutable_observation(AbsolutePos who);
        [[nodiscard]] RoundStage stage() const;
        [[nodiscard]] const Wall *wall() const;
        [[nodiscard]] const Hand *hand(AbsolutePos pos) const;
        Hand *mutable_hand(AbsolutePos pos);
        [[nodiscard]] std::array<const Hand*, 4> hands() const;

        std::string ToMjlog() const;
    private:
        std::uint32_t seed_;
        std::unique_ptr<Score> score_;
        // Round dependent information. These members should be reset after each round.
        RoundStage stage_;
        AbsolutePos dealer_;
        AbsolutePos drawer_;
        std::unique_ptr<Wall> wall_;
        std::array<std::unique_ptr<River>, 4> rivers_;
        std::array<std::unique_ptr<Hand>, 4> hands_;
        std::unique_ptr<ActionHistory> action_history_;

        std::array<std::unique_ptr<Observation>, 4> observations_;

        std::uint32_t GenerateRoundSeed();
        [[nodiscard]] bool NullCheck() const;
    };
}  // namespace mj

#endif //MAHJONG_STATE_H
