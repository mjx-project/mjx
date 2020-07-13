#ifndef MAHJONG_OBSERVATION_H
#define MAHJONG_OBSERVATION_H

#include <mahjong.pb.h>

#include <utility>
#include <array>
#include "hand.h"
#include "action.h"
#include "player.h"

namespace mj
{
    class Score
    {
    public:
        Score();
        [[nodiscard]] std::uint8_t round() const;  // 局
        [[nodiscard]] std::uint8_t honba() const;  // 本場
        [[nodiscard]] std::uint8_t riichi() const;  // リー棒
        [[nodiscard]] std::array<std::int16_t, 4> ten() const;  // 点 250 start
    private:
        friend class Observation;
        mjproto::Score score_{};
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
        PossibleAction(mjproto::PossibleAction possible_action);
        ActionType type() const;
        std::unique_ptr<Open> open() const;
        std::vector<Tile> discard_candidates() const;

        static PossibleAction CreateDiscard(const Hand& hand);
    private:
        friend class Observation;
        mjproto::PossibleAction possible_action_{};
    };

    class ActionHistory
    {
    public:
        ActionHistory() = default;
        [[nodiscard]] std::size_t size() const;
    private:
        friend class Observation;
        mjproto::ActionHistory action_history_{};
    };

    class Observation
    {
    public:
        Observation() = default;
        Observation(AbsolutePos who, Score& score, ActionHistory& action_history, Player& player);
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
        void add_possible_action(PossibleAction possible_action);

        void ClearPossibleActions();
        std::string ToString() const;
    private:
        friend class AgentClient;
        mjproto::ActionRequest action_request_ = mjproto::ActionRequest();
    };
}

#endif //MAHJONG_OBSERVATION_H
