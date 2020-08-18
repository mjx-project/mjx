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
   class Observation
    {
    public:
        Observation() = default;  // Observation is generated only from State::CreatObservation
        // getter
        AbsolutePos who() const;
        Hand initial_hand() const;
        Hand current_hand() const;
        [[nodiscard]] std::vector<PossibleAction> possible_actions() const;
        // setter
        void add_possible_action(PossibleAction &&possible_action);

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
