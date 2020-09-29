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
        Observation() = default;
        ~Observation();

        AbsolutePos who() const;
        [[nodiscard]] bool has_possible_action() const;
        [[nodiscard]] std::vector<PossibleAction> possible_actions() const;
        Hand initial_hand() const;
        Hand current_hand() const;
        std::string ToJson() const;

        void add_possible_action(PossibleAction &&possible_action);
    private:
        friend class AgentClient;
        friend class State;
        Observation(AbsolutePos who, mjproto::State& state);
        mjproto::Observation proto_ = mjproto::Observation{};
    };
}

#endif //MAHJONG_OBSERVATION_H
