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
        Observation(const mjxproto::Observation& proto);

        AbsolutePos who() const;
        [[nodiscard]] bool has_possible_action() const;
        [[nodiscard]] std::vector<mjxproto::Action> possible_actions() const;
        [[nodiscard]] std::vector<Tile> possible_discards() const;
        Hand initial_hand() const;
        Hand current_hand() const;
        std::string ToJson() const;
        const mjxproto::Observation& proto() const;

        void add_possible_action(mjxproto::Action &&possible_action);
        void add_possible_actions(const std::vector<mjxproto::Action> &possible_actions);
    private:
        // TODO: remove friends and use proto()
        friend class State;
        friend class TrainDataGenerator;
        Observation(AbsolutePos who, const mjxproto::State& state);
        mjxproto::Observation proto_ = mjxproto::Observation{};
    };
}

#endif //MAHJONG_OBSERVATION_H
