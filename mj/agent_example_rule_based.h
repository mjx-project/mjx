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
        [[nodiscard]] mjproto::Action TakeAction(Observation &&observation) const final ;
    private:
        template<typename RandomGenerator>
        Tile SelectDiscard(std::vector<Tile> &discard_candidates, const Hand &curr_hand, RandomGenerator& g) const;
    };
}

#endif //MAHJONG_AGENT_EXAMPLE_RULE_BASED_H
