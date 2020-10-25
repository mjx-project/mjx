#ifndef MAHJONG_RULE_BASED_AGENT_H
#define MAHJONG_RULE_BASED_AGENT_H

#include "agent.h"

namespace mj
{
    class RuleBasedAgent final: public Agent
    {
    public:
        RuleBasedAgent() = default;
        RuleBasedAgent(PlayerId player_id);
        ~RuleBasedAgent() final = default;
        [[nodiscard]] Action TakeAction(Observation &&observation) const final ;
    };
}

#endif //MAHJONG_RULE_BASED_AGENT_H
