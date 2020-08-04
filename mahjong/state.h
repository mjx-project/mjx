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
        explicit State(const std::string &json_str);
        bool IsGameOver();

        // operate or access in-round state
        void InitRound();
        bool IsRoundOver();
        AbsolutePos UpdateStateByDraw();
        void UpdateStateByAction(const Action& action);
        Action& UpdateStateByActionCandidates(const std::vector<Action> &action_candidates);
        // operate wall
        Observation CreateObservation(AbsolutePos pos);
        std::optional<std::vector<AbsolutePos>> RonCheck();  // 牌を捨てたプレイヤーの下家から順に
        std::optional<std::vector<std::pair<AbsolutePos, std::vector<Open>>>> StealCheck();

        // accessors
        [[nodiscard]] const Player& player(AbsolutePos pos) const;
        [[nodiscard]] const Wall & wall() const;
        [[nodiscard]] const Hand & hand(AbsolutePos pos) const;
        [[nodiscard]] const River & river(AbsolutePos pos) const;

        std::string ToJson() const;

        static RelativePos ToRelativePos(AbsolutePos origin, AbsolutePos target);
    private:
        std::array<std::string, 4> player_ids_;
        std::uint32_t seed_;
        Score init_score_;
        Score curr_score_;
        // Round dependent information. These members should be reset after each round.
        AbsolutePos last_action_taker_;
        EventType last_event_;
        AbsolutePos dealer_;
        AbsolutePos drawer_;
        AbsolutePos latest_discarder_;
        Wall wall_;
        std::array<Player, 4> players_;

        // protos
        std::array<mjproto::PrivateInfo, 4> private_infos_;
        mjproto::EventHistory event_history_;
        mjproto::Terminal terminal_;

        Player& mutable_player(AbsolutePos pos);

        Tile Draw(AbsolutePos who);
        void Discard(AbsolutePos who, Tile discard);
        void Riichi(AbsolutePos who);
        void ApplyOpen(AbsolutePos who, Open open);
        void AddNewDora();
        void RiichiScoreChange();
        void Tsumo(AbsolutePos winner);
        void Ron(AbsolutePos winner, AbsolutePos loser, Tile tile);
        void NoWinner();

        std::uint32_t GenerateRoundSeed();
    };
}  // namespace mj

#endif //MAHJONG_STATE_H
