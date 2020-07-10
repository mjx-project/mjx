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
#include "player.h"
#include "hand.h"
#include "river.h"
#include "wall.h"

namespace mj
{
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
        [[nodiscard]] const Player& player(AbsolutePos pos) const;
        Player& mutable_player(AbsolutePos pos);
        [[nodiscard]] const Observation& observation(AbsolutePos who) const;
        Observation& mutable_observation(AbsolutePos who);
        [[nodiscard]] RoundStage stage() const;
        [[nodiscard]] const Wall & wall() const;
        [[nodiscard]] const Hand *hand(AbsolutePos pos) const;
        Hand *mutable_hand(AbsolutePos pos);
        [[nodiscard]] std::array<const Hand*, 4> hands() const;

        std::string ToMjlog() const;
    private:
        std::uint32_t seed_;
        Score score_;
        // Round dependent information. These members should be reset after each round.
        RoundStage stage_;
        AbsolutePos dealer_;
        AbsolutePos drawer_;
        Wall wall_;
        std::array<Player, 4> players_;
        ActionHistory action_history_;

        std::array<Observation, 4> observations_;

        std::uint32_t GenerateRoundSeed();
        [[nodiscard]] bool NullCheck() const;
    };
}  // namespace mj

#endif //MAHJONG_STATE_H
