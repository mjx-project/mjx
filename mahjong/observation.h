#ifndef MAHJONG_OBSERVATION_H
#define MAHJONG_OBSERVATION_H

#include <mahjong.pb.h>

#include <utility>
#include <array>
#include "hand.h"
#include "action.h"

namespace mj
{
    class Score
    {
    public:
        Score(): round(0), honba(0), riichi(0), ten({250, 250, 250, 250}) {}
        std::uint8_t round;  // 局
        std::uint8_t honba;  // 本場
        std::uint8_t riichi;  // リー棒
        std::array<std::int16_t, 4> ten;  // 点 250 start
    private:
        friend class Observation;
        std::unique_ptr<mjproto::Score> score_ = std::make_unique<mjproto::Score>();
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
        PossibleAction() = default;
        explicit PossibleAction(mjproto::PossibleAction possible_action);
        ActionType type() const;
        std::unique_ptr<Open> open() const;
        std::vector<Tile> discard_candidates() const;

        static std::unique_ptr<PossibleAction> NewDiscard(const Hand* hand);
    private:
        friend class Observation;
        mjproto::PossibleAction possible_action_;
    };

    class ActionHistory
    {
    public:
        ActionHistory() = default;
        [[nodiscard]] std::size_t size() const;
    private:
        friend class Observation;
        std::unique_ptr<mjproto::ActionHistory> action_history_ = std::make_unique<mjproto::ActionHistory>();
    };

    class Observation
    {
    public:
        Observation() = default;
        Observation(AbsolutePos who, Score* score, ActionHistory* action_history);
        ~Observation();
        // getter
        std::uint32_t game_id() const;
        AbsolutePos who() const;
        Hand initial_hand() const;
        Hand current_hand() const;
        [[nodiscard]] std::vector<PossibleAction> possible_actions() const;
        Score score() const;
        std::vector<TakenAction> taken_actions() const;
        // setter
        void add_possible_action(std::unique_ptr<PossibleAction> possible_action);

        void ClearPossibleActions();
        std::string ToString() const;
    private:
        friend class AgentClient;
        std::unique_ptr<mjproto::ActionRequest> action_request_ = std::make_unique<mjproto::ActionRequest>();
    };
}

#endif //MAHJONG_OBSERVATION_H
