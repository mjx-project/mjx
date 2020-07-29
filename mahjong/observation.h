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
        explicit Score(mjproto::Score score);
        [[nodiscard]] std::uint8_t round() const;  // 局
        [[nodiscard]] std::uint8_t honba() const;  // 本場
        [[nodiscard]] std::uint8_t riichi() const;  // リー棒
        [[nodiscard]] std::array<std::int32_t, 4> ten() const;  // 点 25000 start
    private:
        friend class Observation;  // mjproto::Observation needs to refer to mutable mjproto::Score
        mjproto::Score score_{};
    };

    struct Event {
        AbsolutePos who;
        ActionType type;
        Tile tile;
        Open open;
    };

    class PossibleAction
    {
    public:
        PossibleAction() = default;
        PossibleAction(mjproto::PossibleAction possible_action);
        ActionType type() const;
        Open open() const;
        std::vector<Tile> discard_candidates() const;

        static PossibleAction CreateDiscard(const Hand& hand);
    private:
        friend class Observation;
        mjproto::PossibleAction possible_action_{};
    };

    class Observation
    {
    public:
        Observation() = default;
        Observation(AbsolutePos who, Score& score, Player& player);
        ~Observation();
        // getter
        AbsolutePos who() const;
        Hand initial_hand() const;
        Hand current_hand() const;
        [[nodiscard]] std::vector<PossibleAction> possible_actions() const;
        Score score() const;
        std::vector<Event> taken_actions() const;
        // setter
        void add_possible_action(PossibleAction possible_action);

        void ClearPossibleActions();
        std::string ToString() const;
    private:
        friend class AgentClient;
        mjproto::Observation proto_ = mjproto::Observation{};
    };
}

#endif //MAHJONG_OBSERVATION_H
