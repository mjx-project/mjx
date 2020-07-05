#ifndef MAHJONG_OBSERVATION_H
#define MAHJONG_OBSERVATION_H

#include <mahjong.pb.h>

#include <utility>
#include <array>
#include "hand.h"
#include "action.h"

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

    struct TakenAction {
        AbsolutePos who;
        ActionType type;
        Tile draw;
        Tile discard;
        bool discard_drawn_tile;
        std::unique_ptr<Open> open;
    };

    class PossibleAction
    {
    public:
        explicit PossibleAction(mjproto::ActionRequest_PossibleAction possible_action);
        ActionType type() const;
        std::unique_ptr<Open> open() const;
        std::vector<Tile> discard_candidates() const;
    private:
        friend class Observation;
        mjproto::ActionRequest_PossibleAction possible_action_;
    };

    class CommonObservation
    {
    public:
        CommonObservation() = default;
        // getter
        Score score();
        std::vector<TakenAction> taken_actions();
        // setter
        void set_score(const Score &score);
        void add_taken_action(const TakenAction &taken_action);
    private:
        friend class Observation;
        mjproto::ActionRequest_CommonObservation common_observation_;
    };

    class Observation
    {
    public:
        Observation() = default;
        explicit Observation(AbsolutePos who, CommonObservation& common_observation);
        ~Observation();
        // getter
        std::uint32_t game_id() const;
        AbsolutePos who() const;
        Hand initial_hand() const;
        Hand current_hand() const;
        [[nodiscard]] std::vector<PossibleAction> possible_actions() const;
        Score score() const;
        std::vector<TakenAction> taken_actions() const;
        [[nodiscard]] const mjproto::ActionRequest& action_request() const;
        // setter
        void add_possible_action(std::unique_ptr<PossibleAction> possible_action);

        void ClearPossibleActions();
        std::string ToString() const;
    private:
        mjproto::ActionRequest action_request_;
    };
}

#endif //MAHJONG_OBSERVATION_H
