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
#include "event.h"
#include "player.h"
#include "wall.h"

namespace mj
{
   class State
    {
    public:
        State() = default;
        explicit State(
                std::vector<PlayerId> player_ids,  // 起家, ..., ラス親
                std::uint32_t seed = 9999,
                int round = 0, int honba = 0, int riichi = 0,
                std::array<int, 4> tens = {25000, 25000, 25000, 25000});
        explicit State(const std::string &json_str);
        bool IsRoundOver() const;
        bool IsGameOver() const;
        void Update(std::vector<Action> &&action_candidates);
        std::unordered_map<PlayerId, Observation> CreateObservations() const;
        std::string ToJson() const;
        State Next() const;

        static std::vector<PlayerId> ShufflePlayerIds(std::uint32_t seed, std::vector<PlayerId> player_ids);

        // accessors
        [[nodiscard]] AbsolutePos dealer() const;
        [[nodiscard]] Wind prevalent_wind() const;
        [[nodiscard]] std::uint8_t round() const;  // 局
        [[nodiscard]] std::uint8_t honba() const;  // 本場
        [[nodiscard]] std::uint8_t riichi() const;  // リー棒
        [[nodiscard]] std::int32_t ten(AbsolutePos who) const;  // 点 25000点スタート
        [[nodiscard]] std::array<std::int32_t, 4> tens() const;
        [[nodiscard]] std::uint8_t init_riichi() const;
        [[nodiscard]] std::array<std::int32_t, 4> init_tens() const;
    private:
        // protos
        mjproto::State state_;
        mjproto::Score curr_score_;  // Using state_.terminal.final_score gives wrong serialization when round is not finished.
        // container classes
        Wall wall_;
        std::array<Player, 4> players_;
        // temporal memory
        std::uint32_t seed_;
        Event last_event_;
        TileType last_discard_type_;
        bool is_first_turn_wo_open = true;  // ダブル立直, 九種九牌, 四風連打, etc
        bool is_four_winds = true;
        std::vector<bool> has_nm = {true, true, true, true};
        bool is_round_over_ = false;
        bool require_riichi_score_change_ = false;
        bool require_kan_draw_ = false;
        int require_kan_dora_ = 0;  // 加槓 => 暗槓が続いたときに2回連続でカンドラを開く場合がある https://github.com/sotetsuk/mahjong/issues/199
        std::unordered_map<AbsolutePos, bool> is_ippatsu_;

        // accessors
        [[nodiscard]] const Player& player(AbsolutePos pos) const;
        [[nodiscard]] Player& mutable_player(AbsolutePos pos);
        [[nodiscard]] WinStateInfo win_state_info(AbsolutePos who) const;
        [[nodiscard]] AbsolutePos top_player() const;

        // update
        void Update(Action &&action);

        // event operations
        Tile Draw(AbsolutePos who);
        void Discard(AbsolutePos who, Tile discard);
        void Riichi(AbsolutePos who);
        void ApplyOpen(AbsolutePos who, Open open);
        void AddNewDora();
        void RiichiScoreChange();
        void Tsumo(AbsolutePos winner);
        void Ron(AbsolutePos winner);
        void NoWinner();
        [[nodiscard]] std::unordered_map<PlayerId, Observation> CreateStealAndRonObservation() const;
        [[nodiscard]] std::pair<HandInfo, WinScore> EvalWinHand(AbsolutePos who) const noexcept;

        // utils
        bool IsFourKanNoWinner() const noexcept ;
    };
}  // namespace mj

#endif //MAHJONG_STATE_H
