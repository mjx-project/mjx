#ifndef MAHJONG_OBSERVATION_H
#define MAHJONG_OBSERVATION_H

#include <utility>
#include <array>

#include "mj.pb.h"
#include "hand.h"
#include "action.h"

namespace mj
{
   class Observation
    {
    public:
        Observation() = default;
        Observation(const mjproto::Observation& proto);

        AbsolutePos who() const;
        [[nodiscard]] bool has_possible_action() const;
        [[nodiscard]] std::vector<PossibleAction> possible_actions() const;
        Hand initial_hand() const;
        Hand current_hand() const;
        std::string ToJson() const;
        mjproto::Observation proto() const;

        void add_possible_action(PossibleAction &&possible_action);
    private:
        // TODO: remove friends and use proto()
        friend class State;
        friend class TrainDataGenerator;
        Observation(AbsolutePos who, const mjproto::State& state);
        mjproto::Observation proto_ = mjproto::Observation{};
    };
}

#endif //MAHJONG_OBSERVATION_H
