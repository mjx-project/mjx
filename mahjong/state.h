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
        bool IsGameOver();

        // operate or access in-round state
        void InitRound();
        bool IsRoundOver();
        AbsolutePos UpdateStateByDraw();
        void UpdateStateByAction(const Action& action);
        Action& UpdateStateByActionCandidates(const std::vector<Action> &action_candidates);
        // operate wall
        Observation CreateObservation(AbsolutePos who);
        std::optional<std::vector<AbsolutePos>> RonCheck();  // 牌を捨てたプレイヤーの下家から順に
        std::optional<std::vector<std::pair<AbsolutePos, std::vector<Open>>>> StealCheck();

        std::string ToJson() const;

        static RelativePos ToRelativePos(AbsolutePos origin, AbsolutePos target);
        static Wind ToSeatWind(AbsolutePos who, AbsolutePos dealer);
    private:
        std::array<std::string, 4> player_ids_;
        std::uint32_t seed_;
        AbsolutePos last_action_taker_;
        EventType last_event_;
        AbsolutePos drawer_;
        AbsolutePos latest_discarder_;

        Wall wall_;
        std::array<Player, 4> players_;

        // protos
        mjproto::State state_;
        std::array<mjproto::PrivateInfo, 4> private_infos_;
        mjproto::Terminal terminal_;

        [[nodiscard]] std::uint8_t round() const;  // 局
        [[nodiscard]] std::uint8_t honba() const;  // 本場
        [[nodiscard]] std::uint8_t riichi() const;  // リー棒
        [[nodiscard]] std::int32_t ten(AbsolutePos who) const;  // 点
        [[nodiscard]] std::array<std::int32_t, 4> tens() const;  // 点 25000 start
        [[nodiscard]] AbsolutePos dealer() const;
        [[nodiscard]] Wind prevalent_wind() const;

        Player& mutable_player(AbsolutePos pos);
        const Player& player(AbsolutePos pos) const;

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
