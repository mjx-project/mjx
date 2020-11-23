#ifndef MAHJONG_STATE_H
#define MAHJONG_STATE_H

#include <string>
#include <array>
#include <vector>
#include <random>
#include <utility>
#include "consts.h"
#include "tile.h"
#include "observation.h"
#include "action.h"
#include "event.h"
#include "wall.h"
#include "hand.h"
#include "mj.grpc.pb.h"
#include "utils.h"
#include "yaku_evaluator.h"

namespace mj
{
    // 試合結果（半荘）
    struct GameResult {
        std::uint64_t game_seed;
        std::map<PlayerId, int> rankings;  // 1~4
        std::map<PlayerId, int> tens;  // 25000スタート
    };

    class State
    {
    public:
        struct ScoreInfo
        {
            std::vector<PlayerId> player_ids;  // 起家, ..., ラス親
            std::uint64_t game_seed = 0;
            int round = 0;
            int honba = 0;
            int riichi = 0;
            std::array<int, 4> tens = {25000, 25000, 25000, 25000};
        };
        State() = default;
        explicit State(ScoreInfo score_info);
        explicit State(const std::string &json_str);
        explicit State(const mjproto::State& state);
        bool IsRoundOver() const;
        bool IsGameOver() const;
        void Update(std::vector<mjproto::Action> &&action_candidates);
        std::unordered_map<PlayerId, Observation> CreateObservations() const;
        std::string ToJson() const;
        mjproto::State proto() const;
        GameResult result() const;
        State::ScoreInfo Next() const;

        static std::vector<PlayerId> ShufflePlayerIds(std::uint32_t game_seed, std::vector<PlayerId> player_ids);

        // accessors
        [[nodiscard]] AbsolutePos dealer() const;
        [[nodiscard]] Wind prevalent_wind() const;
        [[nodiscard]] std::uint8_t round() const;  // 局
        [[nodiscard]] std::uint8_t honba() const;  // 本場
        [[nodiscard]] std::uint8_t riichi() const;  // リー棒
        [[nodiscard]] std::uint64_t game_seed() const; // シード値
        [[nodiscard]] std::int32_t ten(AbsolutePos who) const;  // 点 25000点スタート
        [[nodiscard]] std::array<std::int32_t, 4> tens() const;
        [[nodiscard]] std::uint8_t init_riichi() const;
        [[nodiscard]] std::array<std::int32_t, 4> init_tens() const;
        [[nodiscard]] bool HasLastEvent() const;
        [[nodiscard]] const mjproto::Event & LastEvent() const;
        [[nodiscard]] std::optional<Tile> TargetTile() const;   // ロンされうる牌. 直前の捨牌or加槓した牌
        [[nodiscard]] bool IsFirstTurnWithoutOpen() const;
        [[nodiscard]] bool IsFourWinds() const;
        [[nodiscard]] bool IsRobbingKan() const;
        [[nodiscard]] int RequireKanDora() const; // 加槓 => 暗槓が続いたときに2回連続でカンドラを開く場合がある https://github.com/sotetsuk/mahjong/issues/199
        [[nodiscard]] bool RequireKanDraw() const;
        [[nodiscard]] bool RequireRiichiScoreChange() const;

        // comparison
        bool Equals(const State& other) const noexcept ;
        bool CanReach(const State& other) const noexcept ;
   private:
        explicit State(
                std::vector<PlayerId> player_ids,  // 起家, ..., ラス親
                std::uint64_t game_seed = 0,
                int round = 0, int honba = 0, int riichi = 0,
                std::array<int, 4> tens = {25000, 25000, 25000, 25000});

        // Internal structures
        struct Player
        {
            PlayerId player_id;
            AbsolutePos position;
            Hand hand;
            // temporal memory
            std::bitset<34> machi;    // 上がりの形になるための待ち(役の有無を考慮しない). bitsetで管理する
            std::bitset<34> discards; // 今までに捨てた牌のset. bitsetで管理する
            std::bitset<34> missed_tiles = 0;  // 他家の打牌でロンを見逃した牌のbitset. フリテンの判定に使用する.
            bool is_ippatsu = false;
            bool has_nm = true;
        };

        struct HandInfo {
            std::vector<Tile> closed_tiles;
            std::vector<Open> opens;
            std::optional<Tile> win_tile;
        };

        // protos
        mjproto::State state_;
        mjproto::Score curr_score_;  // Using state_.terminal.final_score gives wrong serialization when round is not finished.
        // containers
        Wall wall_;
        std::array<Player, 4> players_;
        // temporal memory
        std::optional<AbsolutePos> three_ronned_player = std::nullopt;

        // accessors
        [[nodiscard]] const Player& player(AbsolutePos pos) const;
        [[nodiscard]] Player& mutable_player(AbsolutePos pos);
        [[nodiscard]] const Hand& hand(AbsolutePos who) const;
        [[nodiscard]] Hand& mutable_hand(AbsolutePos who);
        [[nodiscard]] WinStateInfo win_state_info(AbsolutePos who) const;
        [[nodiscard]] AbsolutePos top_player() const;

        // update
        void Update(mjproto::Action &&action);

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
        std::optional<AbsolutePos> HasPao(AbsolutePos winner) const noexcept ;

        // action validators
        bool CanRon(AbsolutePos who, Tile tile) const;
        bool CanRiichi(AbsolutePos who) const; // デフォルト25000点
        bool CanTsumo(AbsolutePos who) const;

        [[nodiscard]] std::optional<HandInfo> EvalTenpai(AbsolutePos who) const noexcept ;

        static mjproto::State LoadJson(const std::string &json_str) ;

        friend class TrainDataGenerator;
        void UpdateByEvent(const mjproto::Event& event);
    };
}  // namespace mj

#endif //MAHJONG_STATE_H
