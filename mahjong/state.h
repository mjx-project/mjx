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
#include "wall.h"

namespace mj
{
    class State
    {
    public:
        explicit State(std::uint32_t seed = 9999);
        explicit State(const std::string &json_str);
        bool IsRoundOver();
        bool IsGameOver();
        AbsolutePos UpdateStateByDraw();
        void UpdateStateByAction(const Action& action);
        Action& UpdateStateByActionCandidates(const std::vector<Action> &action_candidates);
        Observation CreateObservation(AbsolutePos who);
        std::optional<std::vector<AbsolutePos>> RonCheck();  // 牌を捨てたプレイヤーの下家から順に
        std::optional<std::vector<std::pair<AbsolutePos, std::vector<Open>>>> StealCheck();
        std::string ToJson() const;
    private:
        // protos
        mjproto::State state_;
        mjproto::Score curr_score_;  // Using state_.terminal.final_score gives wrong serialization when round is not finished.
        // container classes
        Wall wall_;
        std::array<Player, 4> players_;
        // temporal memory
        std::uint32_t seed_;
        AbsolutePos last_action_taker_;
        EventType last_event_;
        AbsolutePos drawer_;  // to be removed
        AbsolutePos latest_discarder_;  // to be removed

        // accessors
        [[nodiscard]] std::uint8_t round() const;  // 局
        [[nodiscard]] std::uint8_t honba() const;  // 本場
        [[nodiscard]] std::uint8_t riichi() const;  // リー棒
        [[nodiscard]] std::int32_t ten(AbsolutePos who) const;  // 点
        [[nodiscard]] std::array<std::int32_t, 4> tens() const;  // 点 25000 start
        [[nodiscard]] AbsolutePos dealer() const;
        [[nodiscard]] Wind prevalent_wind() const;
        [[nodiscard]] const Player& player(AbsolutePos pos) const;
        [[nodiscard]] Player& mutable_player(AbsolutePos pos);

        // event operations
        Tile Draw(AbsolutePos who);
        void Discard(AbsolutePos who, Tile discard);
        void Riichi(AbsolutePos who);
        void ApplyOpen(AbsolutePos who, Open open);
        void AddNewDora();
        void RiichiScoreChange();
        void Tsumo(AbsolutePos winner);
        void Ron(AbsolutePos winner, AbsolutePos loser, Tile tile);
        void NoWinner();

        [[nodiscard]] std::pair<HandInfo, WinScore> EvalWinHand(AbsolutePos who) const noexcept;
    };
}  // namespace mj

#endif //MAHJONG_STATE_H
