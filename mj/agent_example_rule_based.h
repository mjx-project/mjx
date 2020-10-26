#ifndef MAHJONG_AGENT_EXAMPLE_RULE_BASED_H
#define MAHJONG_AGENT_EXAMPLE_RULE_BASED_H

#include "agent.h"

namespace mj
{
    // Simple rule-based agent.
    class AgentExampleRuleBased final: public Agent
    {
    public:
        AgentExampleRuleBased() = default;
        explicit AgentExampleRuleBased(PlayerId player_id);
        ~AgentExampleRuleBased() final = default;
        [[nodiscard]] Action TakeAction(Observation &&observation) const final ;
    };
}

#endif //MAHJONG_AGENT_EXAMPLE_RULE_BASED_H
