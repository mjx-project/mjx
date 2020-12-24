#ifndef MAHJONG_AGENT_H
#define MAHJONG_AGENT_H

#include "observation.h"

namespace mj {
    class Agent
    {
    public:
        virtual ~Agent() = default;
        [[nodiscard]] virtual std::vector<mjproto::Action> TakeActions(std::vector<Observation> &&observations) = 0;
        [[nodiscard]] virtual PlayerId player_id() const = 0;
    };
}

#endif //MAHJONG_AGENT_H
