#ifndef MAHJONG_OBSERVATION_H
#define MAHJONG_OBSERVATION_H

#include <utility>
#include <array>

#include "mjx.pb.h"
#include "hand.h"
#include "action.h"

namespace mjx
{
   class Observation
    {
    public:
        Observation() = default;
        Observation(const mjproto::Observation& proto);

        AbsolutePos who() const;
        [[nodiscard]] bool has_possible_action() const;
        [[nodiscard]] std::vector<mjproto::Action> possible_actions() const;
        [[nodiscard]] std::vector<Tile> possible_discards() const;
        Hand initial_hand() const;
        Hand current_hand() const;
        std::string ToJson() const;
        const mjproto::Observation& proto() const;

        void add_possible_action(mjproto::Action &&possible_action);
        void add_possible_actions(const std::vector<mjproto::Action> &possible_actions);
    private:
        // TODO: remove friends and use proto()
        friend class State;
        friend class TrainDataGenerator;
        Observation(AbsolutePos who, const mjproto::State& state);
        mjproto::Observation proto_ = mjproto::Observation{};
    };
}

#endif //MAHJONG_OBSERVATION_H
