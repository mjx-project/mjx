#ifndef MAHJONG_AGENT_EXAMPLE_H
#define MAHJONG_AGENT_EXAMPLE_H

#include "agent.h"

namespace mj
{
    // Simple rule-based agent.
    class AgentExample final: public Agent
    {
    public:
        AgentExample() = default;
        explicit AgentExample(PlayerId player_id);
        ~AgentExample() final = default;
        [[nodiscard]] Action TakeAction(Observation &&observation) const final ;
    };
}

#endif //MAHJONG_AGENT_EXAMPLE_H
