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

    class PossibleAction {
    public:
        explicit PossibleAction(mjproto::ActionRequest_PossibleAction possible_action);
        ActionType type() const;
        std::unique_ptr<Open> open() const;
        std::vector<Tile> discard_candidates() const;
    private:
        mjproto::ActionRequest_PossibleAction possible_action_;
    };

    class Observation
    {
    public:
        // As Observation should return the ownership of common observation. The life span of Observation should be short.
        Observation() = delete;
        Observation(mjproto::ActionRequest &action_request, mjproto::ActionRequest_CommonObservation* common_observation)
        : action_request_(action_request) {
            action_request_.set_allocated_common_observation(common_observation);
        }
        ~Observation() {
            // Calling release_common_observation prevent gRPC from deleting common_observation object
            action_request_.release_common_observation();
        }
        std::uint32_t game_id() const;
        AbsolutePos who() const;
        Hand initial_hand() const;
        Hand current_hand() const;
        [[nodiscard]] std::vector<PossibleAction> possible_actions() const;
        Score score() const;
        std::vector<TakenAction> taken_actions() const;
        [[nodiscard]] const mjproto::ActionRequest& action_request() const { return action_request_; }
        void ClearPossibleActions();
        std::string ToString() const;
    private:
        mjproto::ActionRequest &action_request_;
    };
}

#endif //MAHJONG_OBSERVATION_H
