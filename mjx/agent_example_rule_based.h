#ifndef MAHJONG_AGENT_EXAMPLE_RULE_BASED_H
#define MAHJONG_AGENT_EXAMPLE_RULE_BASED_H

#include "agent.h"
#include "strategy_rule_based.h"

namespace mjx
{
    // Simple rule-based agent.
    class AgentExampleRuleBased final: public Agent
    {
    public:
        AgentExampleRuleBased() = default;
        explicit AgentExampleRuleBased(PlayerId player_id);
        ~AgentExampleRuleBased() final = default;
        [[nodiscard]] mjxproto::Action TakeAction(Observation &&observation) const final ;
    private:
        StrategyRuleBased strategy_;
    };
}

#endif //MAHJONG_AGENT_EXAMPLE_RULE_BASED_H
