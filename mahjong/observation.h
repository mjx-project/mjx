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

        static PossibleAction CreateDiscard(const std::vector<Tile>& possible_discards);
    private:
        friend class Observation;
        mjproto::PossibleAction possible_action_{};
    };

    class Observation
    {
    public:
        Observation() = delete;  // Observation is generated only from State::CreatObservation
        // getter
        AbsolutePos who() const;
        Hand initial_hand() const;
        Hand current_hand() const;
        [[nodiscard]] std::vector<PossibleAction> possible_actions() const;
        std::vector<Event> taken_actions() const;
        // setter
        void add_possible_action(PossibleAction possible_action);

        void ClearPossibleActions();
        std::string ToString() const;
    private:
        friend class AgentClient;
        friend class State;
        Observation(AbsolutePos who, const mjproto::State& state);
        mjproto::Observation proto_ = mjproto::Observation{};
    };
}

#endif //MAHJONG_OBSERVATION_H
